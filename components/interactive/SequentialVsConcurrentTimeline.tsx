"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Timer, Zap, Play, RotateCcw, ArrowRight } from "lucide-react";

const COLORS = ["#ef4444", "#3b82f6", "#10b981"];

export function SequentialVsConcurrentTimeline() {
  const [delays, setDelays] = useState([2, 1, 3]);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const syncTotal = delays.reduce((a, b) => a + b, 0);
  const asyncTotal = Math.max(...delays);
  const speedup = syncTotal / asyncTotal;

  const start = () => {
    setRunning(true);
    setElapsed(0);
    const start = Date.now();
    timerRef.current = setInterval(() => {
      const e = (Date.now() - start) / 1000;
      setElapsed(e);
      if (e >= syncTotal) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const getSyncTaskProgress = (index: number) => {
    const start = delays.slice(0, index).reduce((a, b) => a + b, 0);
    const end = start + delays[index];
    if (elapsed <= start) return 0;
    if (elapsed >= end) return 100;
    return ((elapsed - start) / delays[index]) * 100;
  };

  const getAsyncTaskProgress = (index: number) => {
    if (elapsed <= 0) return 0;
    if (elapsed >= delays[index]) return 100;
    return (elapsed / delays[index]) * 100;
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Timer className="w-5 h-5" />
        顺序执行 vs 并发执行
      </h3>
      <div className="flex items-center gap-4 mb-6">
        {delays.map((d, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-xs font-medium" style={{ color: COLORS[i] }}>任务{String.fromCharCode(65 + i)}:</span>
            <input type="range" min={1} max={5} value={d} disabled={running}
              onChange={(e) => { const n = [...delays]; n[i] = Number(e.target.value); setDelays(n); }}
              className="w-20" />
            <span className="text-xs text-slate-500 w-6">{d}s</span>
          </div>
        ))}
        <button onClick={start} disabled={running} className="ml-auto px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> 开始
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="text-center text-sm text-slate-500 mb-2">已用时间: {elapsed.toFixed(1)}s</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-red-200 dark:border-red-800 p-5">
          <h4 className="font-bold text-red-600 dark:text-red-400 mb-4 flex items-center gap-2">
            <ArrowRight className="w-4 h-4" /> 顺序执行
          </h4>
          <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-4 h-36">
            {delays.map((d, i) => {
              const start = delays.slice(0, i).reduce((a, b) => a + b, 0);
              return (
                <div key={i} className="relative h-7 mb-1" style={{ marginLeft: `${(start / syncTotal) * 100}%` }}>
                  <div className="absolute inset-0 bg-slate-200 dark:bg-slate-700 rounded" style={{ width: `${(d / syncTotal) * 100}%` }} />
                  <motion.div className="absolute inset-0 rounded" style={{ width: `${(d / syncTotal) * 100}%`, backgroundColor: COLORS[i], opacity: 0.8 }}
                    animate={{ clipPath: `inset(0 ${100 - getSyncTaskProgress(i)}% 0 0)` }} />
                  <span className="absolute left-1 top-0.5 text-[10px] text-white font-bold z-10">
                    {String.fromCharCode(65 + i)}({d}s)
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-3 text-sm font-medium text-red-600 dark:text-red-400">
            总耗时: {delays.join(" + ")} = {syncTotal} 秒
          </div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-green-200 dark:border-green-800 p-5">
          <h4 className="font-bold text-green-600 dark:text-green-400 mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4" /> 并发执行
          </h4>
          <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-4 h-36">
            {delays.map((d, i) => (
              <div key={i} className="relative h-7 mb-1">
                <div className="absolute inset-0 bg-slate-200 dark:bg-slate-700 rounded" style={{ width: `${(d / asyncTotal) * 100}%` }} />
                <motion.div className="absolute inset-0 rounded" style={{ width: `${(d / asyncTotal) * 100}%`, backgroundColor: COLORS[i], opacity: 0.8 }}
                  animate={{ clipPath: `inset(0 ${100 - getAsyncTaskProgress(i)}% 0 0)` }} />
                <span className="absolute left-1 top-0.5 text-[10px] text-white font-bold z-10">
                  {String.fromCharCode(65 + i)}({d}s)
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm font-medium text-green-600 dark:text-green-400">
            总耗时: max({delays.join(", ")}) = {asyncTotal} 秒
            <span className="ml-2 text-slate-400">加速: {speedup.toFixed(1)}x</span>
          </div>
        </div>
      </div>
    </div>
  );
}

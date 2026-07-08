"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Pause, Clock, ArrowRight } from "lucide-react";

type StepState = "idle" | "running" | "paused" | "resumed" | "done";

export function FirstAwaitDemo() {
  const [state, setState] = useState<StepState>("idle");
  const [progress, setProgress] = useState(0);
  const [timeLine, setTimeLine] = useState<{ time: number; label: string }[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef(0);

  const stateColors: Record<StepState, string> = {
    idle: "bg-slate-400",
    running: "bg-blue-500",
    paused: "bg-amber-500",
    resumed: "bg-green-500",
    done: "bg-emerald-500",
  };

  const stateLabels: Record<StepState, string> = {
    idle: "就绪",
    running: "运行中",
    paused: "暂停（await sleep）",
    resumed: "恢复执行",
    done: "完成",
  };

  const start = () => {
    setState("running");
    setProgress(0);
    setTimeLine([{ time: 0, label: "开始执行 async def 函数" }]);
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      const totalDuration = 6;

      if (elapsed < 1.5) {
        setProgress((elapsed / totalDuration) * 100);
        setState("running");
      } else if (elapsed < 4.0) {
        setProgress((1.5 / totalDuration) * 100);
        setState("paused");
      } else if (elapsed < 5.5) {
        const resumedProgress = 1.5 + (elapsed - 4.0);
        setProgress((resumedProgress / totalDuration) * 100);
        setState("resumed");
      } else {
        setProgress(100);
        setState("done");
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);

    setTimeout(() => setTimeLine((p) => [...p, { time: 1.5, label: "遇到 await asyncio.sleep(2)" }]), 1500);
    setTimeout(() => setTimeLine((p) => [...p, { time: 4.0, label: "sleep 完成，恢复执行" }]), 4000);
    setTimeout(() => setTimeLine((p) => [...p, { time: 5.5, label: "函数执行完毕，返回结果" }]), 5500);
  };

  const reset = () => {
    setState("idle");
    setProgress(0);
    setTimeLine([]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Clock className="w-5 h-5 text-blue-500" />
        await asyncio.sleep(2) 逐步演示
      </h3>

      <div className="flex gap-3 mb-6">
        <button onClick={start} disabled={state !== "idle" && state !== "done"}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> 开始演示
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Code */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
{`async def demo():
    print("开始执行")          # ← 运行阶段
    await asyncio.sleep(2)     # ← 暂停阶段
    print("执行完成")          # ← 恢复阶段
    return "done"`}
        </pre>
      </div>

      {/* Progress Bar */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">执行进度</span>
          <span className={`px-2 py-1 rounded text-xs font-medium text-white ${stateColors[state]}`}>
            {stateLabels[state]}
          </span>
        </div>
        <div className="w-full h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden relative">
          <motion.div
            className={`h-full rounded-full ${stateColors[state]}`}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />
          {/* Pause indicator */}
          {state === "paused" && (
            <motion.div className="absolute inset-0 flex items-center justify-center"
              animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}>
              <Pause className="w-4 h-4 text-amber-600" />
            </motion.div>
          )}
        </div>

        {/* Timeline markers */}
        <div className="mt-4 flex items-center gap-1">
          <div className="flex-1 relative h-2">
            <div className="absolute left-0 w-[25%] h-2 bg-blue-400 rounded-l" />
            <div className="absolute left-[25%] w-[42%] h-2 bg-amber-400" />
            <div className="absolute left-[67%] w-[23%] h-2 bg-green-400" />
            <div className="absolute left-[90%] w-[10%] h-2 bg-emerald-400 rounded-r" />
          </div>
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
          <span>运行</span>
          <span>暂停 (await)</span>
          <span>恢复</span>
          <span>完成</span>
        </div>
      </div>

      {/* Timeline Events */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
        <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-3">事件时间线</h4>
        <div className="space-y-2">
          {timeLine.map((event, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900 flex items-center justify-center text-xs font-bold text-indigo-600 dark:text-indigo-400">
                {i + 1}
              </div>
              <ArrowRight className="w-4 h-4 text-slate-400" />
              <span className="text-sm text-slate-700 dark:text-slate-300">{event.label}</span>
            </motion.div>
          ))}
          {timeLine.length === 0 && (
            <p className="text-sm text-slate-500">点击"开始演示"查看执行过程</p>
          )}
        </div>
      </div>
    </div>
  );
}

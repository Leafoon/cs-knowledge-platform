"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Users, Play, RotateCcw } from "lucide-react";

const TASKS = [
  { name: "任务A", color: "#ef4444", duration: 3 },
  { name: "任务B", color: "#3b82f6", duration: 2 },
  { name: "任务C", color: "#10b981", duration: 4 },
];

export function ConcurrencyVsParallelism() {
  const [mode, setMode] = useState<"concurrency" | "parallelism">("concurrency");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [taskStates, setTaskStates] = useState<{ progress: number; done: boolean }[]>(
    TASKS.map(() => ({ progress: 0, done: false }))
  );
  const [activeTask, setActiveTask] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setTaskStates(TASKS.map(() => ({ progress: 0, done: false })));
    setActiveTask(0);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      const now = Date.now();
      const e = (now - startTime) / 1000;
      setElapsed(e);

      if (mode === "parallelism") {
        setTaskStates((prev) =>
          prev.map((s, i) => {
            const newProgress = Math.min((e / TASKS[i].duration) * 100, 100);
            return { progress: newProgress, done: newProgress >= 100 };
          })
        );
        if (e >= Math.max(...TASKS.map((t) => t.duration))) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      } else {
        setTaskStates((prev) => {
          const newStates = [...prev];
          let timeLeft = e;
          for (let i = 0; i < newStates.length; i++) {
            if (timeLeft <= 0) break;
            const used = Math.min(timeLeft, TASKS[i].duration);
            newStates[i] = { progress: (used / TASKS[i].duration) * 100, done: used >= TASKS[i].duration };
            timeLeft -= used;
          }
          return newStates;
        });
        setActiveTask(Math.min(Math.floor(e / TASKS[0].duration), TASKS.length - 1));
        if (e >= TASKS.reduce((a, b) => a + b.duration, 0)) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setTaskStates(TASKS.map(() => ({ progress: 0, done: false })));
    setActiveTask(0);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const totalTime = mode === "concurrency" ? TASKS.reduce((a, b) => a + b.duration, 0) : Math.max(...TASKS.map((t) => t.duration));

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Cpu className="w-5 h-5" />
        并发 vs 并行
      </h3>
      <div className="flex gap-3 mb-6">
        <button onClick={() => { setMode("concurrency"); reset(); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${mode === "concurrency" ? "bg-amber-600 text-white" : "bg-slate-100 dark:bg-slate-800"}`}>
          并发（一个厨师）
        </button>
        <button onClick={() => { setMode("parallelism"); reset(); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium ${mode === "parallelism" ? "bg-green-600 text-white" : "bg-slate-100 dark:bg-slate-800"}`}>
          并行（三个厨师）
        </button>
        <button onClick={start} disabled={running}
          className="ml-auto px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
        <div className="flex items-center gap-2 mb-4 text-sm text-slate-500">
          <span>时间: {elapsed.toFixed(1)}s / {totalTime}s</span>
        </div>
        <div className="space-y-3">
          {TASKS.map((task, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="w-12 text-xs font-medium text-slate-600 dark:text-slate-400">{task.name}</span>
              <div className="flex-1 h-8 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden relative">
                <motion.div className="h-full rounded-full flex items-center justify-end pr-2"
                  style={{ width: `${taskStates[i].progress}%`, backgroundColor: task.color }}>
                  {taskStates[i].progress > 10 && <span className="text-[10px] text-white font-bold">{Math.round(taskStates[i].progress)}%</span>}
                </motion.div>
                {mode === "concurrency" && activeTask === i && running && (
                  <motion.div className="absolute inset-0 border-2 rounded-full" style={{ borderColor: task.color }}
                    animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1 }} />
                )}
              </div>
              <span className="w-8 text-xs text-slate-400">{task.duration}s</span>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-slate-600 dark:text-slate-400">
          {mode === "concurrency" ? (
            <p>一个厨师交替做三份菜：某一时刻只执行一个任务，总耗时 = 3 + 2 + 4 = <strong>9 秒</strong></p>
          ) : (
            <p>三个厨师同时做三份菜：三个任务真正并行执行，总耗时 = max(3, 2, 4) = <strong>4 秒</strong></p>
          )}
        </div>
      </div>
    </div>
  );
}

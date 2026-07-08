"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Users, Lock, CheckCircle, Clock, AlertCircle } from "lucide-react";

interface Task { id: number; state: "等待" | "运行中" | "已完成"; startTick: number; duration: number; }

export function SemaphoreDemo() {
  const [maxConcurrent, setMaxConcurrent] = useState(2);
  const [totalTasks] = useState(10);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const reset = () => {
    setRunning(false); setTick(0);
    setTasks(Array.from({ length: totalTasks }, (_, i) => ({ id: i, state: "等待", startTick: -1, duration: Math.floor(Math.random() * 3) + 2 })));
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => { reset(); }, []);

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      setTick((t) => t + 1);
      setTasks((prev) => {
        const next = prev.map((t) => ({ ...t }));
        let active = next.filter((t) => t.state === "运行中").length;
        for (const task of next) { if (task.state === "等待" && active < maxConcurrent) { task.state = "运行中"; task.startTick = tick; active++; } }
        for (const task of next) { if (task.state === "运行中" && tick - task.startTick >= task.duration) { task.state = "已完成"; active--; } }
        return next;
      });
    }, 500);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, tick, maxConcurrent]);

  const waiting = tasks.filter((t) => t.state === "等待").length;
  const active = tasks.filter((t) => t.state === "运行中").length;
  const done = tasks.filter((t) => t.state === "已完成").length;
  const allDone = done === totalTasks && totalTasks > 0;

  useEffect(() => { if (allDone && running) setRunning(false); }, [allDone, running]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Users className="w-5 h-5 text-purple-500" /> 信号量演示 — 并发控制
      </h3>
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <Lock className="w-4 h-4 text-slate-500" />
          <label className="text-sm text-slate-700 dark:text-slate-300">最大并发:</label>
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} onClick={() => { reset(); setMaxConcurrent(n); }}
              className={`w-8 h-8 rounded-lg text-sm font-bold ${n === maxConcurrent ? "bg-purple-600 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}>{n}</button>
          ))}
        </div>
        <button onClick={() => { if (allDone) reset(); setRunning(true); }} disabled={running}
          className="px-4 py-2 rounded-lg bg-purple-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 启动
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[{ v: waiting, l: "等待中", icon: <Clock className="w-4 h-4 mx-auto text-blue-500 mb-1" />, c: "blue" },
          { v: active, l: `运行中 (上限 ${maxConcurrent})`, icon: <AlertCircle className="w-4 h-4 mx-auto text-green-500 mb-1" />, c: "green" },
          { v: done, l: "已完成", icon: <CheckCircle className="w-4 h-4 mx-auto text-slate-500 mb-1" />, c: "slate" }
        ].map(({ v, l, icon, c }) => (
          <div key={l} className={`rounded-lg bg-${c}-50 dark:bg-${c}-900/20 border border-${c}-200 dark:border-${c}-800 p-3 text-center`}>
            {icon}<div className={`text-xl font-bold text-${c}-700 dark:text-${c}-300`}>{v}</div><div className={`text-xs text-${c}-600 dark:text-${c}-400`}>{l}</div>
          </div>
        ))}
      </div>
      <div className="space-y-2">
        {tasks.map((task) => (
          <motion.div key={task.id} layout className="flex items-center gap-3">
            <span className="text-xs font-mono w-16 text-slate-500">任务 {task.id + 1}</span>
            <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden relative">
              <motion.div animate={{ width: task.state === "已完成" ? "100%" : task.state === "运行中" ? `${Math.min(((tick - task.startTick) / task.duration) * 100, 100)}%` : "0%" }}
                className={`h-full rounded-full ${task.state === "已完成" ? "bg-green-400 dark:bg-green-600" : task.state === "运行中" ? "bg-blue-500 dark:bg-blue-600" : "bg-transparent"}`} />
              {task.state === "等待" && <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-400">等待中...</div>}
            </div>
            <span className="text-xs w-12 text-right text-slate-500">{task.duration}s</span>
          </motion.div>
        ))}
      </div>
      {allDone && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-sm text-green-700 dark:text-green-300 flex items-center gap-2">
          <CheckCircle className="w-4 h-4" /> 全部完成! 信号量限制了最多 {maxConcurrent} 个任务同时运行
        </motion.div>
      )}
    </div>
  );
}

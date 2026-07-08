"use client";

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Zap, CheckCircle } from "lucide-react";

interface TaskInfo {
  id: number;
  name: string;
  duration: number;
  color: string;
  progress: number;
  done: boolean;
  completedAt: number | null;
}

const INITIAL: Omit<TaskInfo, "progress" | "done" | "completedAt">[] = [
  { id: 1, name: "IO密集任务", duration: 3000, color: "bg-blue-500" },
  { id: 2, name: "CPU密集任务", duration: 1500, color: "bg-emerald-500" },
  { id: 3, name: "网络请求", duration: 800, color: "bg-amber-500" },
  { id: 4, name: "数据库查询", duration: 2200, color: "bg-rose-500" },
];

export function AsCompletedDemo() {
  const [tasks, setTasks] = useState<TaskInfo[]>(
    INITIAL.map((t) => ({ ...t, progress: 0, done: false, completedAt: null }))
  );
  const [completionOrder, setCompletionOrder] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef(0);

  const start = () => {
    setRunning(true);
    setCompletionOrder([]);
    setTasks(INITIAL.map((t) => ({ ...t, progress: 0, done: false, completedAt: null })));
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      setTasks((prev) => {
        const updated = prev.map((t) => {
          if (t.done) return t;
          const p = Math.min(100, (elapsed / t.duration) * 100);
          return { ...t, progress: p };
        });
        return updated;
      });
    }, 50);

    INITIAL.forEach((t) => {
      setTimeout(() => {
        setTasks((prev) =>
          prev.map((tk) => (tk.id === t.id ? { ...tk, progress: 100, done: true, completedAt: Date.now() - startTimeRef.current } : tk))
        );
        setCompletionOrder((prev) => [...prev, t.id]);
      }, t.duration);
    });

    setTimeout(() => {
      if (timerRef.current) clearInterval(timerRef.current);
      setRunning(false);
    }, Math.max(...INITIAL.map((t) => t.duration)) + 100);
  };

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setRunning(false);
    setCompletionOrder([]);
    setTasks(INITIAL.map((t) => ({ ...t, progress: 0, done: false, completedAt: null })));
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-cyan-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <Zap className="w-7 h-7 text-cyan-600 dark:text-cyan-400" />
        as_completed 演示
      </h3>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={start} disabled={running} className="px-5 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 disabled:opacity-50 dark:bg-cyan-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 开始</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 dark:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="space-y-3 mb-5">
        {tasks.map((t) => (
          <div key={t.id} className="bg-white dark:bg-slate-800 rounded-lg p-3 shadow">
            <div className="flex items-center justify-between mb-1">
              <span className="font-semibold text-slate-700 dark:text-slate-200">{t.name}（{t.duration}ms）</span>
              {t.done && <CheckCircle className="w-5 h-5 text-emerald-500" />}
            </div>
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
              <motion.div className={`h-3 rounded-full ${t.color}`} animate={{ width: `${t.progress}%` }} transition={{ duration: 0.05 }} />
            </div>
            {t.done && <span className="text-xs text-slate-500 dark:text-slate-400 mt-1 inline-block">完成于 {t.completedAt}ms</span>}
          </div>
        ))}
      </div>

      <AnimatePresence>
        {completionOrder.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
            <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">完成顺序（as_completed 结果顺序）</h4>
            <div className="flex gap-2 flex-wrap">
              {completionOrder.map((id, i) => {
                const t = INITIAL.find((x) => x.id === id)!;
                return (
                  <motion.span key={id} initial={{ scale: 0 }} animate={{ scale: 1 }} className={`px-3 py-1 rounded text-white text-sm font-bold ${t.color}`}>
                    #{i + 1} {t.name}
                  </motion.span>
                );
              })}
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">注意：结果按完成时间返回，而非输入顺序</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

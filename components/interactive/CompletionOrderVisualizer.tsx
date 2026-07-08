"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ArrowDown, ArrowUpDown } from "lucide-react";

interface Task {
  id: number;
  name: string;
  delay: number;
  color: string;
  done: boolean;
}

export function CompletionOrderVisualizer() {
  const [tasks] = useState<Task[]>([
    { id: 1, name: "task_1", delay: 3000, color: "bg-blue-500", done: false },
    { id: 2, name: "task_2", delay: 1000, color: "bg-emerald-500", done: false },
    { id: 3, name: "task_3", delay: 2500, color: "bg-amber-500", done: false },
    { id: 4, name: "task_4", delay: 500, color: "bg-rose-500", done: false },
    { id: 5, name: "task_5", delay: 1800, color: "bg-purple-500", done: false },
  ]);
  const [running, setRunning] = useState(false);
  const [doneSet, setDoneSet] = useState<Task[]>([]);
  const [completed, setCompleted] = useState<Set<number>>(new Set());

  const start = () => {
    setRunning(true);
    setDoneSet([]);
    setCompleted(new Set());

    const sorted = [...tasks].sort((a, b) => a.delay - b.delay);
    sorted.forEach((t) => {
      setTimeout(() => {
        setCompleted((prev) => new Set([...prev, t.id]));
        setDoneSet((prev) => [...prev, t]);
      }, t.delay);
    });

    setTimeout(() => setRunning(false), Math.max(...tasks.map((t) => t.delay)) + 200);
  };

  const reset = () => {
    setRunning(false);
    setDoneSet([]);
    setCompleted(new Set());
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-teal-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <ArrowUpDown className="w-7 h-7 text-teal-600 dark:text-teal-400" />
        完成顺序可视化
      </h3>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={start} disabled={running} className="px-5 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 dark:bg-teal-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 开始</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">输入顺序</h4>
          <div className="space-y-2">
            {tasks.map((t, i) => (
              <motion.div key={t.id} animate={{ opacity: completed.has(t.id) ? 1 : 0.5 }} className={`flex items-center gap-2 p-2 rounded ${completed.has(t.id) ? "bg-slate-50 dark:bg-slate-700" : ""}`}>
                <span className="w-6 h-6 flex items-center justify-center bg-slate-200 dark:bg-slate-600 rounded text-xs font-bold text-slate-600 dark:text-slate-300">{i + 1}</span>
                <div className={`w-3 h-3 rounded-full ${t.color}`} />
                <span className="font-mono text-sm text-slate-700 dark:text-slate-200">{t.name}</span>
                <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">{t.delay}ms</span>
                {completed.has(t.id) && <span className="text-emerald-500">✓</span>}
              </motion.div>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-1">
            完成顺序 <ArrowDown className="w-4 h-4 text-teal-500" />
          </h4>
          <div className="space-y-2 min-h-[200px]">
            <AnimatePresence>
              {doneSet.map((t, i) => (
                <motion.div key={t.id} initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="flex items-center gap-2 p-2 bg-emerald-50 dark:bg-emerald-900/30 rounded">
                  <span className="w-6 h-6 flex items-center justify-center bg-emerald-200 dark:bg-emerald-700 rounded text-xs font-bold text-emerald-700 dark:text-emerald-200">{i + 1}</span>
                  <div className={`w-3 h-3 rounded-full ${t.color}`} />
                  <span className="font-mono text-sm text-slate-700 dark:text-slate-200">{t.name}</span>
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 ml-auto">{t.delay}ms</span>
                </motion.div>
              ))}
            </AnimatePresence>
            {doneSet.length === 0 && <p className="text-slate-400 dark:text-slate-500 text-sm text-center pt-8">等待完成...</p>}
          </div>
        </div>
      </div>

      <div className="mt-4 bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p>{`async for coro in asyncio.as_completed(tasks):`}</p>
        <p className="pl-4">{`result = await coro  # 按完成顺序获取结果`}</p>
      </div>
    </div>
  );
}

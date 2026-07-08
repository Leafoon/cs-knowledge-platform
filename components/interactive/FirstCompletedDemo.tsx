"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Trophy, XCircle, RotateCcw } from "lucide-react";

interface RaceTask {
  id: number;
  name: string;
  delay: number;
  status: "pending" | "running" | "won" | "cancelled";
  color: string;
}

export function FirstCompletedDemo() {
  const [tasks, setTasks] = useState<RaceTask[]>([
    { id: 1, name: "服务器A", delay: 3000, status: "pending", color: "bg-blue-500" },
    { id: 2, name: "服务器B", delay: 1000, status: "pending", color: "bg-emerald-500" },
    { id: 3, name: "服务器C", delay: 2000, status: "pending", color: "bg-amber-500" },
  ]);
  const [winner, setWinner] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [positions, setPositions] = useState<number[]>([0, 0, 0]);

  const start = () => {
    setRunning(true);
    setWinner(null);
    setTasks((prev) => prev.map((t) => ({ ...t, status: "running" as const })));
    setPositions([0, 0, 0]);

    const startTime = Date.now();
    const trackLength = 100;
    let settled = false;

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      setPositions((prev) =>
        tasks.map((t, i) => {
          if (prev[i] >= trackLength) return trackLength;
          return Math.min(trackLength, (elapsed / t.delay) * trackLength);
        })
      );
    }, 30);

    tasks.forEach((t) => {
      setTimeout(() => {
        if (settled) return;
        settled = true;
        clearInterval(interval);
        setWinner(t.name);
        setTasks((prev) =>
          prev.map((tk) => (tk.id === t.id ? { ...tk, status: "won" as const } : { ...tk, status: "cancelled" as const }))
        );
        setPositions((prev) => prev.map((p, i) => (tasks[i].id === t.id ? 100 : p)));
        setRunning(false);
      }, t.delay);
    });
  };

  const reset = () => {
    setRunning(false);
    setWinner(null);
    setTasks((prev) => prev.map((t) => ({ ...t, status: "pending" as const })));
    setPositions([0, 0, 0]);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-yellow-50 dark:from-slate-900 dark:to-yellow-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <Trophy className="w-7 h-7 text-yellow-600 dark:text-yellow-400" />
        First Completed 竞争模式
      </h3>

      <p className="text-sm text-slate-600 dark:text-slate-300 text-center mb-4">3个任务竞争，最先完成的获胜，其余被取消</p>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={start} disabled={running} className="px-5 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:opacity-50 dark:bg-yellow-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 开始竞争</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 dark:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow space-y-4 mb-4">
        {tasks.map((t, i) => (
          <div key={t.id}>
            <div className="flex items-center justify-between mb-1">
              <span className="font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-1">
                {t.name} ({t.delay}ms)
                {t.status === "won" && <Trophy className="w-4 h-4 text-yellow-500" />}
                {t.status === "cancelled" && <XCircle className="w-4 h-4 text-red-400" />}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400">{Math.round(positions[i])}%</span>
            </div>
            <div className="relative w-full bg-slate-200 dark:bg-slate-700 rounded-full h-6 overflow-hidden">
              <motion.div
                className={`h-6 rounded-full ${t.color} ${t.status === "cancelled" ? "opacity-40" : ""}`}
                animate={{ width: `${positions[i]}%` }}
                transition={{ duration: 0.03 }}
              />
              <motion.div className="absolute right-1 top-0.5 text-lg" animate={{ scale: t.status === "won" ? [1, 1.3, 1] : 1 }}>
                {t.status === "won" ? "🏆" : t.status === "cancelled" ? "❌" : "🏃"}
              </motion.div>
            </div>
          </div>
        ))}
      </div>

      <AnimatePresence>
        {winner && (
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="bg-yellow-50 dark:bg-yellow-900/30 border-2 border-yellow-400 rounded-lg p-4 text-center">
            <Trophy className="w-8 h-8 text-yellow-600 dark:text-yellow-400 mx-auto mb-2" />
            <p className="font-bold text-lg text-yellow-800 dark:text-yellow-200">{winner} 获胜！</p>
            <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">其他任务已被取消（CancelledError）</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p>{`done, pending = await asyncio.wait(tasks, return_when=FIRST_COMPLETED)`}</p>
        <p className="text-slate-500 dark:text-slate-400"># done: {'{'}获胜任务{'}'}  pending: {'{'}未完成任务{'}'}</p>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, Unlock, Play, RotateCcw, AlertTriangle, Shield } from "lucide-react";

export function LockDemo() {
  const [useLock, setUseLock] = useState(false);
  const [running, setRunning] = useState(false);
  const [counter, setCounter] = useState(0);
  const [step, setStep] = useState(0);
  const [coroutines, setCoroutines] = useState([
    { id: 0, name: "协程 A", pc: 0, local: 0, state: "就绪" },
    { id: 1, name: "协程 B", pc: 0, local: 0, state: "就绪" },
  ]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const steps = useLock ? ["lock()", "load", "add 1", "store", "unlock()"] : ["load", "add 1", "store"];

  const reset = () => {
    setRunning(false); setCounter(0); setStep(0);
    setCoroutines([
      { id: 0, name: "协程 A", pc: 0, local: 0, state: "就绪" },
      { id: 1, name: "协程 B", pc: 0, local: 0, state: "就绪" },
    ]);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      setStep((s) => { if (s >= steps.length * 2) { setRunning(false); return s; } return s + 1; });
      setCoroutines((prev) => prev.map((co) => {
        const ls = step - co.id * steps.length;
        if (ls < 0 || ls >= steps.length) return { ...co, state: step >= (co.id + 1) * steps.length ? "完成" : co.state };
        const next = { ...co, pc: ls, state: "运行中" as const };
        if (steps[ls] === "load") next.local = counter;
        else if (steps[ls] === "add 1") next.local += 1;
        return next;
      }));
    }, 600);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, step, steps.length, counter]);

  useEffect(() => { if (!running && step > 0) setCounter(useLock ? 2 : 1); }, [running, step, useLock]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Lock className="w-5 h-5 text-blue-500" /> 互斥锁演示 — 竞态条件 vs 安全访问
      </h3>
      <div className="flex flex-wrap gap-3 mb-4">
        <button onClick={() => { reset(); setUseLock(!useLock); }}
          className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 ${useLock ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300" : "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"}`}>
          {useLock ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />} {useLock ? "锁: 开启" : "锁: 关闭"}
        </button>
        <button onClick={() => { reset(); setRunning(true); }} disabled={running}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 运行
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {coroutines.map((co) => (
          <motion.div key={co.id} layout className={`rounded-xl border p-4 ${co.state === "运行中" ? "border-blue-400 bg-blue-50 dark:bg-blue-900/20" : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-slate-900 dark:text-slate-100">{co.name}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${co.state === "运行中" ? "bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200" : co.state === "完成" ? "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-300" : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300"}`}>{co.state}</span>
            </div>
            <div className="text-sm text-slate-600 dark:text-slate-400">局部变量: <span className="font-mono font-bold">{co.local}</span></div>
            <div className="flex gap-1 mt-2">{steps.map((_, i) => (<div key={i} className={`h-2 flex-1 rounded-full ${i < co.pc ? "bg-green-400" : i === co.pc ? "bg-blue-500" : "bg-slate-200 dark:bg-slate-600"}`} />))}</div>
            <div className="text-xs mt-1">{co.pc < steps.length ? `当前: ${steps[co.pc]}` : "已完成"}</div>
          </motion.div>
        ))}
      </div>
      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4">
        <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">共享计数器</div>
        <div className="text-4xl font-bold text-slate-900 dark:text-slate-100">{counter}</div>
        {!running && step > 0 && (
          <div className={`mt-2 text-sm flex items-center gap-2 ${counter === 2 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
            {counter === 2 ? <Shield className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
            {counter === 2 ? "正确! 互斥锁防止了竞态条件" : "竞态条件! 两个协程读取了相同的旧值"}
          </div>
        )}
      </div>
    </div>
  );
}

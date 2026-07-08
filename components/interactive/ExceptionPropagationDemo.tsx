"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, AlertTriangle, ArrowUp } from "lucide-react";

const LAYERS = [
  { name: "main()", color: "bg-blue-500", lightBg: "bg-blue-50 dark:bg-blue-900/20", border: "border-blue-300 dark:border-blue-700" },
  { name: "outer_coroutine()", color: "bg-amber-500", lightBg: "bg-amber-50 dark:bg-amber-900/20", border: "border-amber-300 dark:border-amber-700" },
  { name: "inner_coroutine()", color: "bg-rose-500", lightBg: "bg-rose-50 dark:bg-rose-900/20", border: "border-rose-300 dark:border-rose-700" },
];

export function ExceptionPropagationDemo() {
  const [step, setStep] = useState(-1);
  const [running, setRunning] = useState(false);

  const steps = [
    { event: "inner_coroutine 抛出 ValueError", layer: 2, hasError: true },
    { event: "异常向上传播到 outer_coroutine", layer: 1, hasError: true, propagating: true },
    { event: "outer_coroutine 未捕获，继续传播", layer: 1, hasError: true, propagating: true },
    { event: "main() 的 try/except 捕获异常", layer: 0, hasError: true, caught: true },
  ];

  const start = () => {
    setRunning(true);
    setStep(-1);
    steps.forEach((_, i) => {
      setTimeout(() => setStep(i), (i + 1) * 1200);
    });
    setTimeout(() => setRunning(false), steps.length * 1200 + 200);
  };

  const reset = () => {
    setRunning(false);
    setStep(-1);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-rose-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <AlertTriangle className="w-7 h-7 text-rose-600 dark:text-rose-400" />
        异常传播演示
      </h3>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={start} disabled={running} className="px-5 py-2 bg-rose-600 text-white rounded-lg hover:bg-rose-700 disabled:opacity-50 dark:bg-rose-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 演示</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="space-y-3 mb-5">
        {LAYERS.map((layer, i) => {
          const isActive = step >= 0 && steps[step]?.layer === i;
          const isError = step >= 0 && steps[step]?.layer === i && steps[step]?.hasError;
          const isCaught = step >= 0 && steps[step]?.caught && i === 0;

          return (
            <motion.div key={i} animate={{ scale: isActive ? 1.02 : 1 }} style={{ marginLeft: i * 24 }}
              className={`rounded-lg p-4 border-2 ${isActive ? layer.border : "border-slate-200 dark:border-slate-700"} ${isActive ? layer.lightBg : "bg-white dark:bg-slate-800"} shadow`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${layer.color}`} />
                  <span className="font-bold text-slate-800 dark:text-slate-100">{layer.name}</span>
                </div>
                {isError && !isCaught && (
                  <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="flex items-center gap-1 text-rose-600 dark:text-rose-400">
                    <ArrowUp className="w-4 h-4" />
                    <span className="text-xs font-bold">ValueError</span>
                  </motion.div>
                )}
                {isCaught && (
                  <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-xs font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-900/30 px-2 py-1 rounded">
                    try/except 捕获 ✓
                  </motion.span>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      <AnimatePresence>
        {step >= 0 && step < steps.length && (
          <motion.div key={step} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-slate-800 rounded-lg p-3 shadow mb-4">
            <p className="text-sm text-slate-700 dark:text-slate-200">
              <span className="font-bold text-rose-600 dark:text-rose-400">步骤 {step + 1}:</span> {steps[step].event}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p>{`async def main():`}</p>
        <p className="pl-4">{`try:`}</p>
        <p className="pl-8">{`await outer_coroutine()`}</p>
        <p className="pl-4">{`except ValueError as e:`}</p>
        <p className="pl-8">{`print(f"Caught: {e}")  # 捕获来自 inner 的异常`}</p>
      </div>
    </div>
  );
}

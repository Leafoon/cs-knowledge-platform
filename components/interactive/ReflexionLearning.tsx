"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCcw, Check, X } from "lucide-react";

interface Iteration {
  attempt: number;
  action: string;
  result: "success" | "failure";
  reflection: string;
}

const ITERATIONS: Iteration[] = [
  { attempt: 1, action: "直接计算 100/0", result: "failure", reflection: "除以零会报错，需要先检查分母" },
  { attempt: 2, action: "if分母!=0 then 计算", result: "success", reflection: "成功处理了边界情况" },
];

export function ReflexionLearning() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <RotateCcw className="w-6 h-6 text-rose-500" />
        Reflexion 自我反思
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        通过反思错误来学习和改进，每次失败都是一次学习机会。
      </p>

      <div className="flex gap-3 mb-6">
        {ITERATIONS.map((iter, idx) => (
          <button
            key={idx}
            onClick={() => setCurrent(idx)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              current === idx
                ? "bg-rose-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            {iter.result === "success" ? <Check className="w-4 h-4" /> : <X className="w-4 h-4" />}
            尝试 {iter.attempt}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
        >
          <div className="space-y-4">
            <div>
              <span className="text-sm text-slate-500">执行操作</span>
              <p className="font-mono text-slate-700 dark:text-slate-200">{ITERATIONS[current].action}</p>
            </div>
            <div>
              <span className={`text-sm ${ITERATIONS[current].result === "success" ? "text-green-600" : "text-red-600"}`}>
                {ITERATIONS[current].result === "success" ? "成功" : "失败"}
              </span>
            </div>
            <div>
              <span className="text-sm text-slate-500">反思</span>
              <p className="text-rose-600 dark:text-rose-400">{ITERATIONS[current].reflection}</p>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ClipboardList, Play, CheckCircle, RotateCcw, ArrowRight } from "lucide-react";

interface PlanStep {
  id: number;
  task: string;
  status: "pending" | "executing" | "done";
}

const INITIAL_PLAN: PlanStep[] = [
  { id: 1, task: "分析用户需求：订机票从北京到上海", status: "done" },
  { id: 2, task: "搜索可用航班", status: "done" },
  { id: 3, task: "比较价格和时间", status: "executing" },
  { id: 4, task: "选择最佳航班", status: "pending" },
  { id: 5, task: "完成预订", status: "pending" },
];

export function PlanExecuteFlow() {
  const [plan, setPlan] = useState<PlanStep[]>(INITIAL_PLAN);
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = () => {
    if (isRunning) return;
    setIsRunning(true);
    setPlan(INITIAL_PLAN.map((s) => ({ ...s, status: "pending" })));

    let currentStep = 0;
    const interval = setInterval(() => {
      setPlan((prev) =>
        prev.map((step, idx) => ({
          ...step,
          status: idx < currentStep ? "done" : idx === currentStep ? "executing" : "pending",
        }))
      );
      currentStep++;
      if (currentStep > INITIAL_PLAN.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1000);
  };

  const handleReset = () => {
    setPlan(INITIAL_PLAN);
    setIsRunning(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <ClipboardList className="w-6 h-6 text-emerald-500" />
        Plan-and-Execute 流程演示
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        先制定完整计划，然后逐步执行。适合复杂、多步骤的任务。
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          {isRunning ? "执行中..." : "执行计划"}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="space-y-3">
          {plan.map((step, idx) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`flex items-center gap-4 p-4 rounded-lg transition-all ${
                step.status === "done"
                  ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                  : step.status === "executing"
                  ? "bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500"
                  : "bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700"
              }`}
            >
              <span className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                step.status === "done"
                  ? "bg-green-500 text-white"
                  : step.status === "executing"
                  ? "bg-blue-500 text-white animate-pulse"
                  : "bg-slate-200 dark:bg-slate-700 text-slate-500"
              }`}>
                {step.status === "done" ? <CheckCircle className="w-5 h-5" /> : step.id}
              </span>
              <span className={`flex-1 ${
                step.status === "done"
                  ? "text-green-700 dark:text-green-300 line-through opacity-70"
                  : step.status === "executing"
                  ? "text-blue-700 dark:text-blue-300 font-medium"
                  : "text-slate-600 dark:text-slate-400"
              }`}>
                {step.task}
              </span>
              {step.status === "executing" && (
                <ArrowRight className="w-5 h-5 text-blue-500 animate-bounce" />
              )}
            </motion.div>
          ))}
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-500 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full" />
          已完成
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
          执行中
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-slate-300 rounded-full" />
          待执行
        </div>
      </div>
    </div>
  );
}

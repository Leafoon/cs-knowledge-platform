"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, CheckCircle, Clock, Database } from "lucide-react";

interface WALStep {
  id: number;
  label: string;
  labelZh: string;
  description: string;
  journalState: string;
  diskState: string;
  color: string;
}

const walSteps: WALStep[] = [
  {
    id: 1,
    label: "Journal Write",
    labelZh: "日志写入",
    description: "将修改意图写入日志区域。修改的块（数据块、inode、位图）先写入日志区，不修改实际位置。",
    journalState: "块 100、200、300 写入日志区",
    diskState: "实际位置未修改",
    color: "blue",
  },
  {
    id: 2,
    label: "Journal Commit",
    labelZh: "日志提交",
    description: "原子地写入提交标记。这是一个单块写入，原子性由硬件保证。如果提交成功，事务一定可以恢复。",
    journalState: "commit 标记写入日志头",
    diskState: "实际位置仍未修改",
    color: "amber",
  },
  {
    id: 3,
    label: "Checkpoint",
    labelZh: "检查点",
    description: "将日志中的块数据复制到实际磁盘位置。这一步可能涉及多次写入，但如果崩溃，日志已提交可以重放。",
    journalState: "日志数据仍然存在",
    diskState: "块 100、200、300 写入实际位置",
    color: "emerald",
  },
  {
    id: 4,
    label: "Free Log",
    labelZh: "释放日志",
    description: "清除日志头（设置 n=0），释放日志空间供后续事务使用。",
    journalState: "日志清空（n=0）",
    diskState: "所有修改已写入",
    color: "purple",
  },
];

export default function WALFlowVisualizer() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);

  const reset = useCallback(() => {
    setCurrentStep(-1);
    setIsRunning(false);
  }, []);

  const autoPlay = useCallback(() => {
    reset();
    setIsRunning(true);
    let i = 0;
    const interval = setInterval(() => {
      setCurrentStep(i);
      i++;
      if (i >= walSteps.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1500);
  }, [reset]);

  const stepForward = () => {
    if (currentStep < walSteps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const getBlockColor = (step: number, blockType: "journal" | "disk") => {
    if (blockType === "journal") {
      if (step >= 1) return "bg-blue-400 dark:bg-blue-500";
      return "bg-slate-200 dark:bg-gray-700";
    }
    if (step >= 3) return "bg-emerald-400 dark:bg-emerald-500";
    return "bg-slate-200 dark:bg-gray-700";
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        写前日志（WAL）流程
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        日志写 → 日志提交（原子）→ 检查点 → 释放日志
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={autoPlay}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          自动播放
        </button>
        <button
          onClick={stepForward}
          disabled={isRunning || currentStep >= walSteps.length - 1}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          单步
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* Disk state visualization */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-bold text-slate-700 dark:text-gray-200">
              日志区域
            </span>
          </div>
          <div className="flex gap-2">
            {["头块", "块100", "块200", "块300"].map((label, i) => (
              <motion.div
                key={i}
                className={`flex-1 h-12 rounded flex items-center justify-center text-xs font-mono transition-all ${
                  i === 0
                    ? currentStep >= 2
                      ? "bg-amber-200 dark:bg-amber-800 text-amber-700 dark:text-amber-300"
                      : "bg-slate-100 dark:bg-gray-700 text-slate-400"
                    : currentStep >= 1
                    ? "bg-blue-200 dark:bg-blue-800 text-blue-700 dark:text-blue-300"
                    : "bg-slate-100 dark:bg-gray-700 text-slate-400"
                }`}
                animate={
                  currentStep === (i === 0 ? 2 : 1)
                    ? { scale: [1, 1.05, 1] }
                    : {}
                }
              >
                {label}
              </motion.div>
            ))}
          </div>
          {currentStep >= 4 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-2 text-xs text-emerald-600 dark:text-emerald-400 text-center"
            >
              日志已清空
            </motion.div>
          )}
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-4 h-4 text-emerald-500" />
            <span className="text-sm font-bold text-slate-700 dark:text-gray-200">
              实际磁盘位置
            </span>
          </div>
          <div className="flex gap-2">
            {["块100", "块200", "块300"].map((label, i) => (
              <motion.div
                key={i}
                className={`flex-1 h-12 rounded flex items-center justify-center text-xs font-mono transition-all ${
                  currentStep >= 3
                    ? "bg-emerald-200 dark:bg-emerald-800 text-emerald-700 dark:text-emerald-300"
                    : "bg-slate-100 dark:bg-gray-700 text-slate-400"
                }`}
                animate={currentStep === 3 ? { scale: [1, 1.05, 1] } : {}}
              >
                {label}
              </motion.div>
            ))}
          </div>
          {currentStep >= 3 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-2 text-xs text-emerald-600 dark:text-emerald-400 text-center"
            >
              数据已从日志复制到实际位置
            </motion.div>
          )}
        </div>
      </div>

      {/* Step timeline */}
      <div className="space-y-2">
        {walSteps.map((step, i) => {
          const isVisible = i <= currentStep;
          const isCurrent = i === currentStep;

          return (
            <AnimatePresence key={step.id}>
              {isVisible && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    isCurrent
                      ? `border-${step.color}-400 dark:border-${step.color}-500 bg-white dark:bg-gray-800 shadow-md`
                      : "border-slate-200 dark:border-gray-700 bg-white/60 dark:bg-gray-800/60"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                        isCurrent
                          ? `bg-${step.color}-500`
                          : "bg-slate-400 dark:bg-gray-600"
                      }`}
                    >
                      {step.id}
                    </span>
                    <span className="text-sm font-bold text-slate-800 dark:text-gray-100">
                      {step.labelZh}
                    </span>
                    <span className="text-xs font-mono text-slate-500 dark:text-gray-400">
                      {step.label}
                    </span>
                    {isCurrent && (
                      <motion.span
                        className="ml-auto w-2 h-2 rounded-full bg-blue-500"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      />
                    )}
                  </div>
                  {isCurrent && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="mt-3 ml-10 text-sm text-slate-600 dark:text-gray-300 leading-relaxed"
                    >
                      {step.description}
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          );
        })}
      </div>

      {currentStep >= walSteps.length - 1 && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 text-center"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-sm font-bold">
            <CheckCircle className="w-4 h-4" />
            事务完成！在任何崩溃点都可以保证一致性
          </span>
        </motion.div>
      )}
    </div>
  );
}

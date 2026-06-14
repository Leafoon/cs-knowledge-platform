"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, CheckCircle } from "lucide-react";

interface CommitStep {
  id: number;
  label: string;
  labelZh: string;
  description: string;
  diskOps: string[];
  color: string;
}

const commitSteps: CommitStep[] = [
  {
    id: 1,
    label: "write_log()",
    labelZh: "写入日志数据",
    description: "将 log_write() 标记为脏的缓冲区从缓存写入日志区域对应的磁盘块。",
    diskOps: [
      "日志块 1 ← 缓冲区 block[0]",
      "日志块 2 ← 缓冲区 block[1]",
      "日志块 3 ← 缓冲区 block[2]",
    ],
    color: "blue",
  },
  {
    id: 2,
    label: "write_head()",
    labelZh: "写入日志头（提交）",
    description: "将日志头（包含块数 n 和块号映射）写入日志区域的第一个块。这是原子操作——单块写入由硬件保证原子性。",
    diskOps: ["日志头块 ← {n=3, block=[100,200,300]}"],
    color: "amber",
  },
  {
    id: 3,
    label: "install_trans()",
    labelZh: "安装事务",
    description: "将日志区域中每个块的数据复制到其实际磁盘位置。这一步可能涉及多次写入，但即使崩溃也可以从日志重放。",
    diskOps: [
      "block 100 ← 日志块 1",
      "block 200 ← 日志块 2",
      "block 300 ← 日志块 3",
    ],
    color: "emerald",
  },
  {
    id: 4,
    label: "write_head() (clear)",
    labelZh: "清空日志头",
    description: "将日志头的 n 设置为 0 并写入磁盘，表示日志已释放。此后日志空间可被新事务复用。",
    diskOps: ["日志头块 ← {n=0, block=[]}"],
    color: "purple",
  },
];

export default function Xv6CommitFlow() {
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
      if (i >= commitSteps.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1500);
  }, [reset]);

  const stepForward = () => {
    if (currentStep < commitSteps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        xv6 commit() 流程
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        end_op() 调用 commit() 提交日志事务的四个步骤
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={autoPlay}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-violet-500 text-white rounded-lg hover:bg-violet-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          自动播放
        </button>
        <button
          onClick={stepForward}
          disabled={isRunning || currentStep >= commitSteps.length - 1}
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

      {/* Visual disk state */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            日志区域
          </h3>
          <div className="space-y-1.5">
            {["日志头 (n=3, blocks=[100,200,300])", "日志块 1", "日志块 2", "日志块 3"].map(
              (label, i) => {
                const active =
                  i === 0
                    ? currentStep >= 1
                    : currentStep >= 0;
                const cleared = i === 0 && currentStep >= 3;
                return (
                  <motion.div
                    key={i}
                    className={`h-8 rounded flex items-center px-3 text-xs font-mono transition-all ${
                      cleared
                        ? "bg-slate-100 dark:bg-gray-700 text-slate-400"
                        : active
                        ? i === 0
                          ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                          : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                        : "bg-slate-50 dark:bg-gray-750 text-slate-400"
                    }`}
                    animate={
                      (i === 0 && currentStep === 1) ||
                      (i === 0 && currentStep === 3) ||
                      (i > 0 && currentStep === 0)
                        ? { scale: [1, 1.03, 1] }
                        : {}
                    }
                  >
                    {cleared ? "n=0 (已清空)" : label}
                  </motion.div>
                );
              }
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            实际磁盘位置
          </h3>
          <div className="space-y-1.5">
            {["block 100", "block 200", "block 300"].map((label, i) => {
              const active = currentStep >= 2;
              return (
                <motion.div
                  key={i}
                  className={`h-8 rounded flex items-center px-3 text-xs font-mono transition-all ${
                    active
                      ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300"
                      : "bg-slate-50 dark:bg-gray-750 text-slate-400"
                  }`}
                  animate={currentStep === 2 ? { scale: [1, 1.03, 1] } : {}}
                >
                  {label} {active && "← 已更新"}
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {commitSteps.map((step, i) => {
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
                        className="ml-auto w-2 h-2 rounded-full bg-violet-500"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      />
                    )}
                  </div>
                  {isCurrent && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="mt-3 ml-10"
                    >
                      <p className="text-sm text-slate-600 dark:text-gray-300 leading-relaxed mb-2">
                        {step.description}
                      </p>
                      <div className="space-y-1">
                        {step.diskOps.map((op, j) => (
                          <div
                            key={j}
                            className="text-xs font-mono text-slate-500 dark:text-gray-400 bg-slate-50 dark:bg-gray-750 rounded px-2 py-1"
                          >
                            {op}
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          );
        })}
      </div>

      {currentStep >= commitSteps.length - 1 && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 text-center"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-sm font-bold">
            <CheckCircle className="w-4 h-4" />
            事务已提交！崩溃后可通过日志恢复
          </span>
        </motion.div>
      )}
    </div>
  );
}

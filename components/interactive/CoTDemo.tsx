"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, ArrowRight, Check, RotateCcw } from "lucide-react";

interface ReasoningStep {
  id: number;
  thought: string;
  highlight: boolean;
}

const EXAMPLE_STEPS: ReasoningStep[] = [
  { id: 1, thought: "理解问题：小明有100元，买了3本书每本25元，还剩多少钱？", highlight: false },
  { id: 2, thought: "识别关键信息：初始金额100元，购买3本书，每本25元", highlight: false },
  { id: 3, thought: "制定计划：需要计算总花费，然后用初始金额减去花费", highlight: false },
  { id: 4, thought: "计算总花费：3 × 25 = 75元", highlight: true },
  { id: 5, thought: "计算剩余：100 - 75 = 25元", highlight: true },
  { id: 6, thought: "得出结论：小明还剩25元", highlight: false },
];

export function CoTDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setCurrentStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setCurrentStep(step);
      if (step >= EXAMPLE_STEPS.length - 1) {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 1200);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Brain className="w-6 h-6 text-purple-500" />
        Chain-of-Thought 推理演示
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        通过逐步推理来解决复杂问题。每一步都基于前一步的结论继续思考。
      </p>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handlePlay}
          disabled={isPlaying}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
        >
          <ArrowRight className="w-4 h-4" />
          {isPlaying ? "推理中..." : "开始推理"}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* 推理步骤 */}
      <div className="space-y-3">
        {EXAMPLE_STEPS.map((step, idx) => (
          <AnimatePresence key={step.id}>
            {idx <= currentStep && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`p-4 rounded-xl border-l-4 ${
                  step.highlight
                    ? "bg-purple-50 dark:bg-purple-900/20 border-purple-500"
                    : idx === currentStep
                    ? "bg-blue-50 dark:bg-blue-900/20 border-blue-500"
                    : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700"
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold ${
                    step.highlight
                      ? "bg-purple-500 text-white"
                      : idx === currentStep
                      ? "bg-blue-500 text-white"
                      : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                  }`}>
                    {idx < currentStep ? <Check className="w-4 h-4" /> : step.id}
                  </span>
                  <div>
                    <span className="text-xs font-medium text-slate-500 mb-1 block">
                      {step.highlight ? "关键计算" : "思考步骤"}
                    </span>
                    <p className="text-slate-700 dark:text-slate-200">{step.thought}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        ))}
      </div>

      {/* 进度指示器 */}
      <div className="mt-6 flex items-center gap-2">
        {EXAMPLE_STEPS.map((_, idx) => (
          <div
            key={idx}
            className={`h-2 flex-1 rounded-full transition-all ${
              idx <= currentStep ? "bg-purple-500" : "bg-slate-200 dark:bg-slate-700"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

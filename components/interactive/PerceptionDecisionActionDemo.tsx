"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eye, Brain, Hand, ArrowRight } from "lucide-react";

const STEPS = [
  { id: "perception", name: "感知", icon: Eye, color: "blue", description: "接收用户输入和环境信息" },
  { id: "decision", name: "决策", icon: Brain, color: "purple", description: "LLM 分析并规划行动" },
  { id: "action", name: "执行", icon: Hand, color: "green", description: "调用工具完成操作" },
];

export function PerceptionDecisionActionDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setCurrentStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step = (step + 1) % 3;
      setCurrentStep(step);
      if (step === 0) {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 1500);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        感知-决策-执行循环
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        Agent 的核心行为模式：感知环境 → 做出决策 → 执行行动 → 再次感知。
      </p>

      <button
        onClick={handlePlay}
        disabled={isPlaying}
        className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 mb-6"
      >
        {isPlaying ? "循环中..." : "播放循环"}
      </button>

      <div className="flex items-center justify-center gap-4">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = currentStep === idx;
          return (
            <React.Fragment key={step.id}>
              <motion.div
                animate={{ scale: isActive ? 1.1 : 1 }}
                className={`w-32 h-32 rounded-2xl flex flex-col items-center justify-center transition-all ${
                  isActive
                    ? `bg-${step.color}-100 dark:bg-${step.color}-900/30 border-2 border-${step.color}-500 shadow-lg`
                    : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
                }`}
              >
                <Icon className={`w-10 h-10 mb-2 ${isActive ? `text-${step.color}-500` : "text-slate-400"}`} />
                <span className="font-bold text-slate-800 dark:text-slate-100">{step.name}</span>
              </motion.div>
              {idx < STEPS.length - 1 && (
                <ArrowRight className={`w-8 h-8 ${idx === currentStep ? "text-indigo-500" : "text-slate-300"}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 text-center"
        >
          <span className="font-bold text-slate-800 dark:text-slate-100">
            当前阶段: {STEPS[currentStep].name}
          </span>
          <p className="text-slate-600 dark:text-slate-300 mt-1">{STEPS[currentStep].description}</p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

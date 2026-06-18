"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Wrench, Eye } from "lucide-react";

const CYCLE_STEPS = [
  { id: "thought", name: "Thought", icon: MessageSquare, color: "purple", description: "分析问题，决定下一步" },
  { id: "action", name: "Action", icon: Wrench, color: "blue", description: "选择并调用工具" },
  { id: "observation", name: "Observation", icon: Eye, color: "green", description: "获取工具返回结果" },
];

export function ReactPatternDemo() {
  const [current, setCurrent] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const handlePlay = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    let step = 0;
    const interval = setInterval(() => {
      step = (step + 1) % 3;
      setCurrent(step);
      if (step === 0) {
        clearInterval(interval);
        setIsPlaying(false);
      }
    }, 1200);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">ReAct 循环模式</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        ReAct = Reasoning + Acting。通过 Thought → Action → Observation 的循环解决问题。
      </p>

      <button
        onClick={handlePlay}
        disabled={isPlaying}
        className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700 disabled:opacity-50 mb-6"
      >
        {isPlaying ? "循环中..." : "播放循环"}
      </button>

      <div className="flex items-center justify-center gap-6">
        {CYCLE_STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = current === idx;
          return (
            <motion.div
              key={step.id}
              animate={{ scale: isActive ? 1.1 : 1, opacity: isActive ? 1 : 0.5 }}
              className="text-center"
            >
              <div className={`w-24 h-24 rounded-full flex items-center justify-center mb-2 ${
                isActive
                  ? `bg-${step.color}-100 dark:bg-${step.color}-900/30 border-2 border-${step.color}-500`
                  : "bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
              }`}>
                <Icon className={`w-10 h-10 ${isActive ? `text-${step.color}-500` : "text-slate-400"}`} />
              </div>
              <span className="font-bold text-slate-800 dark:text-slate-100">{step.name}</span>
              <p className="text-xs text-slate-500 mt-1">{step.description}</p>
            </motion.div>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700"
        >
          <span className="font-bold text-pink-600 dark:text-pink-400">示例: </span>
          <span className="text-slate-700 dark:text-slate-200">
            {current === 0 && "用户问'北京天气如何'，我需要搜索天气信息..."}
            {current === 1 && "调用 search_weather(city='北京') 工具..."}
            {current === 2 && "获取结果：北京 28°C，晴天，湿度 45%"}
          </span>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

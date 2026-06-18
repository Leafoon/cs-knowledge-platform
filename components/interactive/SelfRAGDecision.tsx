"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HelpCircle, Check, RefreshCw, MessageSquare } from "lucide-react";

interface Scenario {
  question: string;
  decision: string;
  action: string;
}

const SCENARIOS: Scenario[] = [
  { question: "1+1等于几？", decision: "简单问题，无需检索", action: "直接回答: 2" },
  { question: "2024年诺贝尔奖得主？", decision: "需要最新信息", action: "检索 → 生成" },
  { question: "写一首诗", decision: "创意任务，无需检索", action: "直接生成" },
];

export function SelfRAGDecision() {
  const [current, setCurrent] = useState(0);
  const scenario = SCENARIOS[current];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <HelpCircle className="w-6 h-6 text-emerald-500" />
        Self-RAG 决策演示
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        Self-RAG 让模型自己判断是否需要检索，避免不必要的开销。
      </p>

      <div className="flex gap-2 mb-6">
        {SCENARIOS.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setCurrent(idx)}
            className={`px-4 py-2 rounded-lg transition-all ${
              current === idx
                ? "bg-emerald-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            场景 {idx + 1}
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
          <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-900 rounded-lg">
            <span className="text-sm text-slate-500">用户问题</span>
            <p className="font-medium text-slate-800 dark:text-slate-100">{scenario.question}</p>
          </div>

          <div className="mb-4 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg">
            <span className="text-sm text-emerald-600 flex items-center gap-1">
              <Check className="w-4 h-4" /> 模型判断
            </span>
            <p className="text-emerald-700 dark:text-emerald-200">{scenario.decision}</p>
          </div>

          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <span className="text-sm text-blue-600 flex items-center gap-1">
              <MessageSquare className="w-4 h-4" /> 执行动作
            </span>
            <p className="text-blue-700 dark:text-blue-200">{scenario.action}</p>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

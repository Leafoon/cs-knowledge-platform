"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Check, X, Lightbulb } from "lucide-react";

const PRINCIPLES = [
  { id: "single", name: "单一职责", good: "search_weather(city)", bad: "do_everything(query)", description: "每个工具只做一件事" },
  { id: "clear", name: "自描述性", good: "calculate_bmi(weight, height)", bad: "calc(x, y)", description: "名称清晰传达功能" },
  { id: "idempotent", name: "幂等性", good: "get_user(id) → 不变", bad: "increment_counter()", description: "相同输入产生相同输出" },
  { id: "safe", name: "容错性", good: "try/except + 友好错误", bad: "直接崩溃", description: "优雅处理异常" },
];

export function ToolDesignPrinciples() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Lightbulb className="w-6 h-6 text-emerald-500" />
        工具设计原则
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {PRINCIPLES.map((p, idx) => (
          <button
            key={p.id}
            onClick={() => setSelected(idx)}
            className={`p-3 rounded-xl text-left transition-all ${
              selected === idx
                ? "bg-emerald-100 dark:bg-emerald-900/30 border-2 border-emerald-500"
                : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
            }`}
          >
            <span className="font-bold text-slate-800 dark:text-slate-100 text-sm">{p.name}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
          {PRINCIPLES[selected].name}
        </h4>
        <p className="text-slate-600 dark:text-slate-300 mb-4">{PRINCIPLES[selected].description}</p>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Check className="w-5 h-5 text-green-500" />
              <span className="font-medium text-green-700 dark:text-green-300">正确示例</span>
            </div>
            <code className="text-sm text-green-600 dark:text-green-400">{PRINCIPLES[selected].good}</code>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <X className="w-5 h-5 text-red-500" />
              <span className="font-medium text-red-700 dark:text-red-300">错误示例</span>
            </div>
            <code className="text-sm text-red-600 dark:text-red-400">{PRINCIPLES[selected].bad}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

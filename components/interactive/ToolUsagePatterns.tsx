"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wrench, ArrowRight, CheckCircle, XCircle } from "lucide-react";

interface ToolPattern {
  id: string;
  name: string;
  description: string;
  example: string;
  pros: string[];
  cons: string[];
}

const PATTERNS: ToolPattern[] = [
  {
    id: "single",
    name: "单工具调用",
    description: "每次只调用一个工具，简单直接",
    example: "search(query) → result",
    pros: ["实现简单", "易于调试", "错误处理简单"],
    cons: ["效率低", "无法并行"],
  },
  {
    id: "parallel",
    name: "并行工具调用",
    description: "同时调用多个独立工具，提高效率",
    example: "[search(q1), search(q2)] → [r1, r2]",
    pros: ["效率高", "减少延迟"],
    cons: ["需要处理依赖关系", "错误处理复杂"],
  },
  {
    id: "sequential",
    name: "顺序链式调用",
    description: "前一个工具的输出作为后一个的输入",
    example: "search → analyze → summarize",
    pros: ["适合复杂流程", "数据传递清晰"],
    cons: ["延迟累加", "一个失败全失败"],
  },
  {
    id: "conditional",
    name: "条件分支调用",
    description: "根据条件选择不同的工具",
    example: "if type=='code' then execute else search",
    pros: ["灵活", "智能路由"],
    cons: ["条件判断复杂", "难以测试"],
  },
];

export function ToolUsagePatterns() {
  const [selectedPattern, setSelectedPattern] = useState<string>("single");
  const pattern = PATTERNS.find((p) => p.id === selectedPattern)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Wrench className="w-6 h-6 text-purple-500" />
        工具调用模式
      </h3>

      {/* 模式选择器 */}
      <div className="flex flex-wrap gap-2 mb-6">
        {PATTERNS.map((p) => (
          <button
            key={p.id}
            onClick={() => setSelectedPattern(p.id)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selectedPattern === p.id
                ? "bg-purple-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-purple-100 dark:hover:bg-purple-900/30"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* 模式详情 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={pattern.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
        >
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
            {pattern.name}
          </h4>
          <p className="text-slate-600 dark:text-slate-300 mb-4">
            {pattern.description}
          </p>

          {/* 示例代码 */}
          <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-4 mb-4 font-mono text-sm">
            <span className="text-purple-600 dark:text-purple-400">示例: </span>
            {pattern.example}
          </div>

          {/* 优缺点对比 */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2 flex items-center gap-1">
                <CheckCircle className="w-4 h-4" /> 优点
              </h5>
              <ul className="space-y-1">
                {pattern.pros.map((pro, i) => (
                  <li key={i} className="text-sm text-slate-600 dark:text-slate-300">
                    • {pro}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h5 className="font-semibold text-red-600 dark:text-red-400 mb-2 flex items-center gap-1">
                <XCircle className="w-4 h-4" /> 缺点
              </h5>
              <ul className="space-y-1">
                {pattern.cons.map((con, i) => (
                  <li key={i} className="text-sm text-slate-600 dark:text-slate-300">
                    • {con}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

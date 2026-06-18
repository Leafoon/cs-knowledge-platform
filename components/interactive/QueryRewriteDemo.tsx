"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Edit3, ArrowRight } from "lucide-react";

const EXAMPLES = [
  { original: "什么是机器学习？", rewritten: "机器学习的定义和基本概念是什么？" },
  { original: "Python好学吗？", rewritten: "Python编程语言的学习难度和入门建议" },
  { original: "怎么减肥？", rewritten: "科学有效的减肥方法和健康建议" },
];

export function QueryRewriteDemo() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Edit3 className="w-6 h-6 text-amber-500" />
        查询改写演示
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        将用户的口语化查询改写为更精确、更适合检索的形式。
      </p>

      <div className="flex gap-2 mb-6">
        {EXAMPLES.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setCurrent(idx)}
            className={`px-4 py-2 rounded-lg transition-all ${
              current === idx
                ? "bg-amber-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            示例 {idx + 1}
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
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <span className="text-xs text-slate-500 block mb-1">原始查询</span>
              <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg text-slate-700 dark:text-slate-200">
                {EXAMPLES[current].original}
              </div>
            </div>
            <ArrowRight className="w-8 h-8 text-amber-500 flex-shrink-0" />
            <div className="flex-1">
              <span className="text-xs text-amber-600 block mb-1">改写后</span>
              <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg text-amber-700 dark:text-amber-200">
                {EXAMPLES[current].rewritten}
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

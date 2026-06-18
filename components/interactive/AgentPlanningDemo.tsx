"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Target, ArrowRight } from "lucide-react";

const STEPS = [
  { id: 1, task: "分析问题", status: "done" },
  { id: 2, task: "制定计划", status: "current" },
  { id: 3, task: "执行计划", status: "pending" },
  { id: 4, task: "评估结果", status: "pending" },
];

export function AgentPlanningDemo() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 规划演示</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 4)}
        className="px-4 py-2 bg-amber-600 text-white rounded-lg mb-6">下一步</button>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="space-y-3">
          {STEPS.map((s, i) => (
            <div key={s.id} className={`flex items-center gap-3 p-3 rounded-lg ${i === current ? "bg-amber-50 dark:bg-amber-900/20 border border-amber-300" : i < current ? "bg-green-50 dark:bg-green-900/20" : "bg-slate-50 dark:bg-slate-900"}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${i < current ? "bg-green-500 text-white" : i === current ? "bg-amber-500 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-500"}`}>
                {i < current ? "✓" : s.id}
              </div>
              <span className="font-medium text-slate-800 dark:text-slate-100">{s.task}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

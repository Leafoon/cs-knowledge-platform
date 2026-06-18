"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, BarChart3, FileText } from "lucide-react";

const STEPS = [
  { id: 1, name: "数据加载", icon: Database, description: "读取数据源" },
  { id: 2, name: "数据分析", icon: BarChart3, description: "统计与可视化" },
  { id: 3, name: "报告生成", icon: FileText, description: "输出分析报告" },
];

export function DataAnalysisPipeline() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">数据分析 Agent 流程</h3>

      <button onClick={() => setCurrent((c) => (c + 1) % STEPS.length)}
        className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 mb-6">下一步</button>

      <div className="flex items-center justify-between">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.id} className={`text-center ${idx === current ? "scale-110" : "opacity-50"}`}>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-2 ${idx === current ? "bg-purple-100 dark:bg-purple-900/30 border-2 border-purple-500" : "bg-slate-100 dark:bg-slate-800"}`}>
                <Icon className={`w-8 h-8 ${idx === current ? "text-purple-500" : "text-slate-400"}`} />
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{step.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

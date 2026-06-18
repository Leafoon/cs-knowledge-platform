"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Eye, Brain, Hand, ArrowRight } from "lucide-react";

const STEPS = ["感知环境", "分析决策", "执行行动"];

export function PDAFlowChartV2() {
  const [current, setCurrent] = useState(0);
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">PDA循环V2</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 3)} className="px-4 py-2 bg-indigo-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-4">
        {[Eye, Brain, Hand].map((Icon, i) => (
          <React.Fragment key={i}>
            <div className={`w-24 h-24 rounded-full flex items-center justify-center ${i === current ? "bg-indigo-100 dark:bg-indigo-900/30 border-2 border-indigo-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
              <Icon className={`w-10 h-10 ${i === current ? "text-indigo-500" : "text-slate-400"}`} />
            </div>
            {i < 2 && <ArrowRight className="w-6 h-6 text-slate-300" />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

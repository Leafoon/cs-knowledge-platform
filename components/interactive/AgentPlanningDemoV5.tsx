"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Map, CheckCircle, Circle } from "lucide-react";

const PLANS = [
  { id: 1, task: "数据收集", status: "done" },
  { id: 2, task: "数据分析", status: "current" },
  { id: 3, task: "报告生成", status: "pending" },
];

export function AgentPlanningDemoV5() {
  const [current, setCurrent] = useState(1);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent规划V3</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 3)} className="px-4 py-2 bg-rose-600 text-white rounded-lg mb-6">下一步</button>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="space-y-3">
          {PLANS.map((p, i) => (
            <div key={p.id} className={`flex items-center gap-3 p-3 rounded-lg ${i === current ? "bg-rose-50 dark:bg-rose-900/20 border border-rose-300" : i < current ? "bg-green-50 dark:bg-green-900/20" : "bg-slate-50 dark:bg-slate-900"}`}>
              {i < current ? <CheckCircle className="w-5 h-5 text-green-500"/> : <Circle className="w-5 h-5 text-slate-400"/>}
              <span className="font-medium text-slate-800 dark:text-slate-100">{p.task}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

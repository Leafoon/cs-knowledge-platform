"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { DollarSign, TrendingDown, Zap } from "lucide-react";

const STRATEGIES = [
  { id: 1, name: "模型路由", savings: "50-70%", icon: Zap, description: "根据任务选择模型" },
  { id: 2, name: "Prompt缓存", savings: "30-50%", icon: TrendingDown, description: "缓存常见查询" },
  { id: 3, name: "批处理", savings: "50%", icon: DollarSign, description: "批量API调用" },
];

export function CostOptimizationStrategies() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">成本优化策略</h3>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {STRATEGIES.map((s, idx) => (
          <button key={s.id} onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${selected === idx ? "bg-green-100 dark:bg-green-900/30 border-2 border-green-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <s.icon className={`w-8 h-8 mb-2 ${selected === idx ? "text-green-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{s.name}</span>
            <span className="text-green-600 font-bold">{s.savings}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{STRATEGIES[selected].description}</p>
      </div>
    </div>
  );
}

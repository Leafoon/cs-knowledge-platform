"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Lightbulb, Sparkles, Target } from "lucide-react";

const STRATEGIES = [
  { id: "direct", name: "直接回答", icon: Lightbulb, description: "简单问题直接回答", latency: "低" },
  { id: "tree", name: "树状搜索", icon: Sparkles, description: "探索多条推理路径", latency: "高" },
  { id: "planned", name: "计划驱动", icon: Target, description: "先制定计划再执行", latency: "中" },
];

export function ReasoningStrategySelectorV9() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">推理策略选择V3</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {STRATEGIES.map((s, i) => (
          <button key={s.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <s.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-violet-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{s.name}</span>
            <span className="text-xs text-slate-500">延迟: {s.latency}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{STRATEGIES[selected].description}</p>
      </div>
    </div>
  );
}

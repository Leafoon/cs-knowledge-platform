"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Layers, GitBranch, RotateCcw } from "lucide-react";

const ALGORITHMS = [
  { id: "cot", name: "Chain-of-Thought", icon: Layers, description: "线性推理链", complexity: "低" },
  { id: "tot", name: "Tree of Thoughts", icon: GitBranch, description: "树状搜索", complexity: "高" },
  { id: "reflexion", name: "Reflexion", icon: RotateCcw, description: "反思迭代", complexity: "中" },
];

export function PlanningAlgorithmComparison() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">规划算法对比</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {ALGORITHMS.map((a, i) => (
          <button key={a.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-amber-100 dark:bg-amber-900/30 border-2 border-amber-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <a.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-amber-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{a.name}</span>
            <span className="text-xs text-slate-500">复杂度: {a.complexity}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{ALGORITHMS[selected].description}</p>
      </div>
    </div>
  );
}

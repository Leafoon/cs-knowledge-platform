"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Award, BarChart2, TrendingUp } from "lucide-react";

const BENCHMARKS = [
  { id: "swebench", name: "SWE-bench", score: 72, description: "代码修复能力" },
  { id: "webarena", name: "WebArena", score: 45, description: "网页操作能力" },
  { id: "gaia", name: "GAIA", score: 65, description: "综合推理能力" },
];

export function EvaluationBenchmarkDemo() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Award className="w-6 h-6 text-amber-500" />
        Agent 评估基准
      </h3>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {BENCHMARKS.map((b, idx) => (
          <button key={b.id} onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${selected === idx ? "bg-amber-100 dark:bg-amber-900/30 border-2 border-amber-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <Award className={`w-8 h-8 mb-2 ${selected === idx ? "text-amber-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{b.name}</span>
            <span className="text-xs text-slate-500">{b.description}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{BENCHMARKS[selected].name}</h4>
        <div className="flex items-center gap-4">
          <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-4">
            <motion.div className="h-4 bg-amber-500 rounded-full" initial={{ width: 0 }} animate={{ width: `${BENCHMARKS[selected].score}%` }} />
          </div>
          <span className="font-bold text-amber-600">{BENCHMARKS[selected].score}%</span>
        </div>
      </div>
    </div>
  );
}

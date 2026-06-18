"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Zap, Brain } from "lucide-react";

const STRATEGIES = [
  { id: "zero", name: "Zero-Shot", icon: Zap, description: "无需示例直接推理", cost: "低", accuracy: "中" },
  { id: "few", name: "Few-Shot", icon: Layers, description: "提供少量示例", cost: "中", accuracy: "高" },
  { id: "cot", name: "Chain-of-Thought", icon: Brain, description: "逐步推理", cost: "高", accuracy: "最高" },
];

export function ReasoningStrategySelectorV2() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">推理策略选择器</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {STRATEGIES.map((s, i) => (
          <button key={s.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <s.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-blue-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{s.name}</span>
            <span className="text-xs text-slate-500">成本: {s.cost} | 准确率: {s.accuracy}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{STRATEGIES[selected].description}</p>
      </div>
    </div>
  );
}

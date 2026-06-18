"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Box, Layers, GitBranch } from "lucide-react";

const ARCHITECTURES = [
  { id: "single", name: "单Agent", icon: Box, description: "单一Agent独立完成任务", pros: "简单", cons: "能力有限" },
  { id: "multi", name: "多Agent", icon: Layers, description: "多个Agent协作", pros: "能力强", cons: "协调复杂" },
  { id: "hierarchical", name: "分层架构", icon: GitBranch, description: "分层管理", pros: "可扩展", cons: "延迟高" },
];

export function AgentArchitectureComparisonV2() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 架构对比</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {ARCHITECTURES.map((a, i) => (
          <button key={a.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-purple-100 dark:bg-purple-900/30 border-2 border-purple-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <a.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-purple-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{a.name}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{ARCHITECTURES[selected].description}</p>
        <div className="flex gap-4 mt-3">
          <span className="text-green-600">优点: {ARCHITECTURES[selected].pros}</span>
          <span className="text-red-600">缺点: {ARCHITECTURES[selected].cons}</span>
        </div>
      </div>
    </div>
  );
}

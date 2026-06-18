"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { AlignLeft, Layers, GitMerge } from "lucide-react";

const STRATEGIES = [
  { id: "fixed", name: "固定大小", icon: AlignLeft, description: "按字符数或 token 数分割", pros: "简单高效", cons: "可能切断语义" },
  { id: "recursive", name: "递归分割", icon: Layers, description: "按优先级递归分割", pros: "保持语义完整", cons: "实现复杂" },
  { id: "semantic", name: "语义分块", icon: GitMerge, description: "基于段落/章节边界", pros: "最佳语义", cons: "计算成本高" },
];

export function ChunkingStrategies() {
  const [selected, setSelected] = useState("fixed");

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">文本分块策略</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        分块是 RAG 的关键步骤，直接影响检索质量。
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {STRATEGIES.map((s) => {
          const Icon = s.icon;
          return (
            <button
              key={s.id}
              onClick={() => setSelected(s.id)}
              className={`p-4 rounded-xl text-left transition-all ${
                selected === s.id
                  ? "bg-teal-100 dark:bg-teal-900/30 border-2 border-teal-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
              }`}
            >
              <Icon className={`w-6 h-6 mb-2 ${selected === s.id ? "text-teal-500" : "text-slate-400"}`} />
              <h4 className="font-bold text-slate-800 dark:text-slate-100">{s.name}</h4>
              <p className="text-xs text-slate-500 mt-1">{s.description}</p>
            </button>
          );
        })}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
            <span className="text-sm font-medium text-green-700 dark:text-green-300">优点: </span>
            <span className="text-sm text-green-600 dark:text-green-200">
              {STRATEGIES.find((s) => s.id === selected)?.pros}
            </span>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
            <span className="text-sm font-medium text-red-700 dark:text-red-300">缺点: </span>
            <span className="text-sm text-red-600 dark:text-red-200">
              {STRATEGIES.find((s) => s.id === selected)?.cons}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

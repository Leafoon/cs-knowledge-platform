"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch, ChevronRight } from "lucide-react";

const NODES = [
  { id: 1, question: "需要外部信息？", yes: 2, no: 3 },
  { id: 2, action: "调用工具" },
  { id: 3, question: "需要推理？", yes: 4, no: 5 },
  { id: 4, action: "Chain-of-Thought" },
  { id: 5, action: "直接回答" },
];

export function AgentDecisionTree() {
  const [current, setCurrent] = useState(1);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 决策树</h3>
      <button onClick={() => setCurrent(current === 1 ? 3 : current === 3 ? 5 : 1)}
        className="px-4 py-2 bg-violet-600 text-white rounded-lg mb-6">下一步</button>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          <GitBranch className="w-8 h-8 text-violet-500" />
          <div>
            <span className="font-bold text-slate-800 dark:text-slate-100">
              {NODES.find(n => n.id === current)?.question || NODES.find(n => n.id === current)?.action}
            </span>
            {NODES.find(n => n.id === current)?.action && (
              <span className="ml-2 px-2 py-1 bg-green-100 text-green-700 rounded text-sm">执行</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Box, ArrowRight } from "lucide-react";

const PATTERNS = [
  { id: "loop", name: "循环式", nodes: ["输入", "处理", "输出"] },
  { id: "state", name: "状态机", nodes: ["状态A", "状态B", "状态C"] },
  { id: "reactive", name: "反应式", nodes: ["事件", "处理", "响应"] },
];

export function ArchitecturePatternDemo() {
  const [selected, setSelected] = useState(0);
  const pattern = PATTERNS[selected];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 架构模式</h3>
      <div className="flex gap-3 mb-6">
        {PATTERNS.map((p, i) => (
          <button key={p.id} onClick={() => setSelected(i)}
            className={`px-4 py-2 rounded-lg transition-all ${selected === i ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"}`}>
            {p.name}
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-center gap-4">
          {pattern.nodes.map((n, i) => (
            <React.Fragment key={i}>
              <div className="px-6 py-4 bg-blue-100 dark:bg-blue-900/30 rounded-xl">
                <span className="font-medium text-blue-700 dark:text-blue-300">{n}</span>
              </div>
              {i < pattern.nodes.length - 1 && <ArrowRight className="w-6 h-6 text-blue-400" />}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

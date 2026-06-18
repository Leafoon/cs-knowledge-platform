"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Network, Workflow } from "lucide-react";

const ARCHS = [
  { id: "chain", name: "链式架构", icon: Workflow, desc: "顺序执行任务流" },
  { id: "graph", name: "图架构", icon: Network, desc: "支持循环和分支" },
  { id: "layered", name: "分层架构", icon: Layers, desc: "模块化分层设计" },
];

export function AgentArchitectureComparisonV6() {
  const [selected, setSelected] = useState(0);
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">架构对比V3</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {ARCHS.map((a, i) => (
          <button key={a.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-sky-100 dark:bg-sky-900/30 border-2 border-sky-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <a.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-sky-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{a.name}</span>
            <span className="text-xs text-slate-500">{a.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

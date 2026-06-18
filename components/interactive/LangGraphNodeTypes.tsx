"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Circle, Square, Diamond } from "lucide-react";

const TYPES = [
  { id: "start", name: "开始节点", icon: Circle, color: "green" },
  { id: "process", name: "处理节点", icon: Square, color: "blue" },
  { id: "condition", name: "条件节点", icon: Diamond, color: "amber" },
  { id: "end", name: "结束节点", icon: Circle, color: "red" },
];

export function LangGraphNodeTypes() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">LangGraph 节点类型</h3>
      <div className="flex gap-3 mb-6">
        {TYPES.map((t, i) => (
          <button key={t.id} onClick={() => setSelected(i)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg ${selected === i ? `bg-${t.color}-600 text-white` : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"}`}>
            <t.icon className="w-4 h-4" />
            {t.name}
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          {React.createElement(TYPES[selected].icon, { className: `w-12 h-12 text-${TYPES[selected].color}-500` })}
          <div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{TYPES[selected].name}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

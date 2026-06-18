"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code, Search, Bug, Check } from "lucide-react";

const STEPS = [
  { name: "理解需求", icon: Search, color: "blue" },
  { name: "生成代码", icon: Code, color: "green" },
  { name: "测试修复", icon: Bug, color: "yellow" },
  { name: "完成", icon: Check, color: "green" },
];

export function CodeGenAgentSteps() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">代码生成 Agent 步骤</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 4)}
        className="px-4 py-2 bg-green-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-between">
        {STEPS.map((s, i) => {
          const Icon = s.icon;
          return (
            <div key={i} className={`text-center ${i === current ? "scale-110" : "opacity-50"}`}>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-2 ${i === current ? `bg-${s.color}-100 border-2 border-${s.color}-500` : "bg-slate-100 dark:bg-slate-800"}`}>
                <Icon className={`w-8 h-8 ${i === current ? `text-${s.color}-500` : "text-slate-400"}`} />
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{s.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

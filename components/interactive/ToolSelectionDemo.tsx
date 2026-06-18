"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Wrench, ArrowRight } from "lucide-react";

const TOOLS = [
  { id: "search", name: "搜索工具", description: "搜索互联网信息", icon: "🔍" },
  { id: "calc", name: "计算工具", description: "执行数学计算", icon: "🧮" },
  { id: "code", name: "代码工具", description: "执行Python代码", icon: "💻" },
];

export function ToolSelectionDemo() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具选择演示</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {TOOLS.map((t, i) => (
          <button key={t.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-green-100 dark:bg-green-900/30 border-2 border-green-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <span className="text-3xl mb-2 block">{t.icon}</span>
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{t.name}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{TOOLS[selected].description}</p>
      </div>
    </div>
  );
}

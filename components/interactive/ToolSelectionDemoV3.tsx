"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, Globe, Code } from "lucide-react";

const TOOLS = [
  { id: "db", name: "数据库查询", icon: Database, description: "查询SQL数据库" },
  { id: "web", name: "网络搜索", icon: Globe, description: "搜索互联网信息" },
  { id: "exec", name: "代码执行", icon: Code, description: "执行Python代码" },
];

export function ToolSelectionDemoV3() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具选择V3</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {TOOLS.map((t, i) => (
          <button key={t.id} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-emerald-100 dark:bg-emerald-900/30 border-2 border-emerald-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <t.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-emerald-500" : "text-slate-400"}`} />
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

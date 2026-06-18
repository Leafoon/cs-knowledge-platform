"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Globe, MousePointer, FileText } from "lucide-react";

const CAPABILITIES = [
  { name: "页面导航", icon: Globe, description: "自动浏览网页" },
  { name: "元素交互", icon: MousePointer, description: "点击、输入、提交" },
  { name: "数据提取", icon: FileText, description: "抓取页面内容" },
];

export function WebAgentCapabilities() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Web Agent 能力</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {CAPABILITIES.map((c, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`p-4 rounded-xl transition-all ${selected === i ? "bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <c.icon className={`w-8 h-8 mb-2 ${selected === i ? "text-orange-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{c.name}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{CAPABILITIES[selected].description}</p>
      </div>
    </div>
  );
}

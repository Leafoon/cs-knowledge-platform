"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Globe, MousePointer, FileSearch } from "lucide-react";

const STEPS = [
  { id: 1, name: "页面理解", icon: FileSearch, description: "解析DOM结构" },
  { id: 2, name: "元素定位", icon: MousePointer, description: "定位目标元素" },
  { id: 3, name: "执行操作", icon: Globe, description: "点击、输入、提交" },
];

export function WebAgentFlow() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Web Agent 工作流程</h3>

      <button onClick={() => setCurrent((c) => (c + 1) % STEPS.length)}
        className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 mb-6">下一步</button>

      <div className="flex items-center justify-between">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.id} className={`text-center ${idx === current ? "scale-110" : "opacity-50"}`}>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-2 ${idx === current ? "bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500" : "bg-slate-100 dark:bg-slate-800"}`}>
                <Icon className={`w-8 h-8 ${idx === current ? "text-orange-500" : "text-slate-400"}`} />
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{step.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

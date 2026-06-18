"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Shield, Lock, Eye } from "lucide-react";

const LAYERS = [
  { id: 1, name: "输入验证", icon: Shield, description: "过滤恶意输入" },
  { id: 2, name: "权限控制", icon: Lock, description: "限制工具访问" },
  { id: 3, name: "输出审计", icon: Eye, description: "记录所有操作" },
];

export function AgentSecurityLayersV2() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 安全防护层</h3>

      <button onClick={() => setCurrent((c) => (c + 1) % LAYERS.length)}
        className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 mb-6">下一层</button>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          {React.createElement(LAYERS[current].icon, { className: "w-12 h-12 text-red-500" })}
          <div>
            <span className="font-bold text-slate-800 dark:text-slate-100 text-xl">{LAYERS[current].name}</span>
            <p className="text-slate-600 dark:text-slate-300">{LAYERS[current].description}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

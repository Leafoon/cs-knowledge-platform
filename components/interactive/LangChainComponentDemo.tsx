"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Link, ArrowRight } from "lucide-react";

const COMPONENTS = [
  { name: "PromptTemplate", description: "提示词模板" },
  { name: "OutputParser", description: "输出解析器" },
  { name: "Memory", description: "记忆模块" },
];

export function LangChainComponentDemo() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">LangChain 组件</h3>
      <div className="flex gap-3 mb-6">
        {COMPONENTS.map((c, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`px-4 py-2 rounded-lg ${selected === i ? "bg-teal-600 text-white" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"}`}>
            {c.name}
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100">{COMPONENTS[selected].name}</h4>
        <p className="text-slate-600 dark:text-slate-300">{COMPONENTS[selected].description}</p>
      </div>
    </div>
  );
}

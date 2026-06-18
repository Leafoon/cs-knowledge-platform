"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { MessageSquare, Wrench, Eye, ArrowRight } from "lucide-react";

const STEPS = [
  { type: "Thought", icon: MessageSquare, content: "分析用户需求", color: "purple" },
  { type: "Action", icon: Wrench, content: "调用工具", color: "blue" },
  { type: "Observation", icon: Eye, content: "获取结果", color: "green" },
];

export function ReActDemo() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">ReAct 循环演示</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 3)}
        className="px-4 py-2 bg-pink-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-6">
        {STEPS.map((s, i) => {
          const Icon = s.icon;
          return (
            <React.Fragment key={i}>
              <motion.div animate={{ scale: current === i ? 1.1 : 0.9, opacity: current === i ? 1 : 0.5 }}
                className={`w-28 h-28 rounded-2xl flex flex-col items-center justify-center ${current === i ? `bg-${s.color}-100 dark:bg-${s.color}-900/30 border-2 border-${s.color}-500` : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
                <Icon className={`w-8 h-8 mb-2 ${current === i ? `text-${s.color}-500` : "text-slate-400"}`} />
                <span className="font-bold text-sm text-slate-800 dark:text-slate-100">{s.type}</span>
              </motion.div>
              {i < 2 && <ArrowRight className="w-6 h-6 text-slate-300" />}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Ear, Lightbulb, Zap, ArrowRight } from "lucide-react";

const STEPS = ["感知输入", "思维处理", "行动输出"];

export function PDAFlowChartV4() {
  const [current, setCurrent] = useState(0);
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">PDA循环V4</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 3)} className="px-4 py-2 bg-teal-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-6">
        {[Ear, Lightbulb, Zap].map((Icon, i) => (
          <React.Fragment key={i}>
            <motion.div animate={{ scale: current === i ? 1.1 : 0.9, opacity: current === i ? 1 : 0.5 }}
              className={`w-28 h-28 rounded-2xl flex flex-col items-center justify-center ${current === i ? "bg-teal-100 dark:bg-teal-900/30 border-2 border-teal-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
              <Icon className={`w-10 h-10 mb-2 ${current === i ? "text-teal-500" : "text-slate-400"}`} />
              <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{STEPS[i]}</span>
            </motion.div>
            {i < 2 && <ArrowRight className="w-6 h-6 text-slate-300" />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

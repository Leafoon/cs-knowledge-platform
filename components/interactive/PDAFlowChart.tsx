"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Eye, Brain, Hand, ArrowRight } from "lucide-react";

export function PDAFlowChart() {
  const [step, setStep] = useState(0);
  const steps = [
    { icon: Eye, name: "感知", color: "blue" },
    { icon: Brain, name: "决策", color: "purple" },
    { icon: Hand, name: "执行", color: "green" },
  ];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">PDA 循环流程图</h3>
      <button onClick={() => setStep((s) => (s + 1) % 3)} className="px-4 py-2 bg-indigo-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-4">
        {steps.map((s, i) => {
          const Icon = s.icon;
          return (
            <React.Fragment key={i}>
              <motion.div animate={{ scale: step === i ? 1.2 : 1 }} className={`w-24 h-24 rounded-full flex items-center justify-center ${step === i ? `bg-${s.color}-100 border-2 border-${s.color}-500` : "bg-slate-100 dark:bg-slate-800"}`}>
                <Icon className={`w-10 h-10 ${step === i ? `text-${s.color}-500` : "text-slate-400"}`} />
              </motion.div>
              {i < 2 && <ArrowRight className="w-8 h-8 text-slate-300" />}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

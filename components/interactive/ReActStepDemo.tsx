"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Wrench, Eye, ArrowRight } from "lucide-react";

const STEPS = [
  { type: "Thought", content: "用户想查天气，我需要调用天气工具", icon: MessageSquare, color: "purple" },
  { type: "Action", content: "get_weather(city='北京')", icon: Wrench, color: "blue" },
  { type: "Observation", content: "北京: 28°C, 晴天", icon: Eye, color: "green" },
];

export function ReActStepDemo() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">ReAct 步骤演示</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 3)}
        className="px-4 py-2 bg-pink-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center gap-4">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={i}>
              <motion.div animate={{ scale: current === i ? 1.1 : 0.9, opacity: current === i ? 1 : 0.5 }}
                className={`w-32 h-32 rounded-2xl flex flex-col items-center justify-center ${current === i ? `bg-${step.color}-100 dark:bg-${step.color}-900/30 border-2 border-${step.color}-500` : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
                <Icon className={`w-8 h-8 mb-2 ${current === i ? `text-${step.color}-500` : "text-slate-400"}`} />
                <span className="font-bold text-sm text-slate-800 dark:text-slate-100">{step.type}</span>
              </motion.div>
              {i < 2 && <ArrowRight className="w-6 h-6 text-slate-300" />}
            </React.Fragment>
          );
        })}
      </div>
      <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-700 dark:text-slate-200">{STEPS[current].content}</p>
      </div>
    </div>
  );
}

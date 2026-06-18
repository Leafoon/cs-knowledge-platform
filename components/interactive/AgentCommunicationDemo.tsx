"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { MessageCircle, ArrowRight } from "lucide-react";

const MESSAGES = [
  { from: "Agent A", to: "Agent B", content: "请求数据" },
  { from: "Agent B", to: "Agent A", content: "返回结果" },
];

export function AgentCommunicationDemo() {
  const [step, setStep] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 间通信</h3>
      <button onClick={() => setStep((s) => (s + 1) % 2)}
        className="px-4 py-2 bg-sky-600 text-white rounded-lg mb-6">下一步</button>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-center gap-4">
          <div className="w-24 h-24 rounded-full bg-sky-100 dark:bg-sky-900/30 flex items-center justify-center">
            <MessageCircle className="w-8 h-8 text-sky-500" />
          </div>
          <div className="text-center">
            <ArrowRight className="w-8 h-8 text-sky-400 mx-auto" />
            <span className="text-sm text-slate-500">{MESSAGES[step].content}</span>
          </div>
          <div className="w-24 h-24 rounded-full bg-sky-100 dark:bg-sky-900/30 flex items-center justify-center">
            <MessageCircle className="w-8 h-8 text-sky-500" />
          </div>
        </div>
      </div>
    </div>
  );
}

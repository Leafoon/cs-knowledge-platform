"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, User, Bot } from "lucide-react";

interface HandoffScenario {
  from: string;
  to: string;
  reason: string;
}

const SCENARIOS: HandoffScenario[] = [
  { from: "通用助手", to: "技术专家", reason: "遇到技术问题" },
  { from: "技术专家", to: "客服代表", reason: "需要处理退款" },
  { from: "客服代表", to: "人工客服", reason: "复杂投诉" },
];

export function OpenAIAgentsHandoff() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <ArrowRight className="w-6 h-6 text-purple-500" />
        Agent Handoff 机制
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        当任务超出当前Agent能力时，自动交接给更专业的Agent。
      </p>

      <div className="flex gap-2 mb-6">
        {SCENARIOS.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setCurrent(idx)}
            className={`px-4 py-2 rounded-lg transition-all ${
              current === idx
                ? "bg-purple-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            场景 {idx + 1}
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-center gap-4">
          <div className="text-center">
            <div className="w-16 h-16 bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
              <Bot className="w-8 h-8 text-purple-500" />
            </div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{SCENARIOS[current].from}</span>
          </div>
          <div className="text-center">
            <ArrowRight className="w-8 h-8 text-purple-500 mx-auto mb-1" />
            <span className="text-xs text-slate-500">{SCENARIOS[current].reason}</span>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-2">
              <User className="w-8 h-8 text-green-500" />
            </div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{SCENARIOS[current].to}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

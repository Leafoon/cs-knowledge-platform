"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Clock, ArrowRight } from "lucide-react";

const TIMELINE = [
  { time: "0-30秒", type: "感觉记忆", description: "API原始响应缓冲", duration: "极短" },
  { time: "30秒-5分钟", type: "短期记忆", description: "当前对话上下文", duration: "短" },
  { time: "5分钟+", type: "工作记忆", description: "任务执行中间状态", duration: "中" },
  { time: "永久", type: "长期记忆", description: "持久化重要信息", duration: "长" },
];

export function AgentMemoryTimeline() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 记忆时间线</h3>
      <div className="flex items-center gap-2 mb-6">
        {TIMELINE.map((t, i) => (
          <React.Fragment key={i}>
            <button onClick={() => setSelected(i)}
              className={`px-3 py-2 rounded-lg text-sm transition-all ${selected === i ? "bg-orange-100 dark:bg-orange-900/30 border border-orange-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
              {t.type}
            </button>
            {i < 3 && <ArrowRight className="w-4 h-4 text-slate-300" />}
          </React.Fragment>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100">{TIMELINE[selected].type}</h4>
        <p className="text-slate-600 dark:text-slate-300">{TIMELINE[selected].description}</p>
        <p className="text-sm text-orange-600 mt-2">持续时间: {TIMELINE[selected].duration}</p>
      </div>
    </div>
  );
}

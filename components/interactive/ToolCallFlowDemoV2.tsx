"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Zap, ArrowRight } from "lucide-react";

const STEPS = ["意图识别", "参数提取", "工具调用", "结果处理"];

export function ToolCallFlowDemoV2() {
  const [current, setCurrent] = useState(0);
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具调用流程V2</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 4)} className="px-4 py-2 bg-cyan-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-4">
        {STEPS.map((s, i) => (
          <React.Fragment key={i}>
            <div className={`w-24 h-24 rounded-xl flex items-center justify-center ${i === current ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
              <Zap className={`w-8 h-8 ${i === current ? "text-cyan-500" : "text-slate-400"}`} />
            </div>
            {i < 3 && <ArrowRight className="w-6 h-6 text-slate-300" />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

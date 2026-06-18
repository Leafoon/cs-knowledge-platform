"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Wrench, ArrowRight } from "lucide-react";

const FLOW = [
  { step: "解析意图", description: "理解用户需要调用工具" },
  { step: "选择工具", description: "从工具列表中选择" },
  { step: "生成参数", description: "生成工具调用参数" },
  { step: "执行调用", description: "调用工具获取结果" },
];

export function ToolCallFlowDemo() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具调用流程</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 4)}
        className="px-4 py-2 bg-teal-600 text-white rounded-lg mb-6">下一步</button>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          {FLOW.map((f, i) => (
            <React.Fragment key={i}>
              <div className={`text-center ${i === current ? "scale-110" : "opacity-50"}`}>
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-2 ${i === current ? "bg-teal-100 dark:bg-teal-900/30 border-2 border-teal-500" : "bg-slate-100 dark:bg-slate-800"}`}>
                  <Wrench className={`w-8 h-8 ${i === current ? "text-teal-500" : "text-slate-400"}`} />
                </div>
                <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{f.step}</span>
              </div>
              {i < 3 && <ArrowRight className="w-6 h-6 text-slate-300" />}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

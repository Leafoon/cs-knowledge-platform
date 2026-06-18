"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Send, ArrowRight } from "lucide-react";

const STEPS = ["用户输入", "意图解析", "参数生成", "工具调用", "返回结果"];

export function ToolCallFlowV3() {
  const [current, setCurrent] = useState(0);
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具调用流程V3</h3>
      <button onClick={() => setCurrent((c) => (c + 1) % 5)} className="px-4 py-2 bg-orange-600 text-white rounded-lg mb-6">下一步</button>
      <div className="flex items-center justify-center gap-3">
        {STEPS.map((s, i) => (
          <React.Fragment key={i}>
            <div className={`text-center ${i === current ? "scale-110" : "opacity-50"}`}>
              <div className={`w-20 h-20 rounded-xl flex items-center justify-center mx-auto mb-1 ${i === current ? "bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
                <Send className={`w-6 h-6 ${i === current ? "text-orange-500" : "text-slate-400"}`} />
              </div>
              <span className="text-xs font-medium text-slate-800 dark:text-slate-100">{s}</span>
            </div>
            {i < 4 && <ArrowRight className="w-4 h-4 text-slate-300 flex-shrink-0" />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

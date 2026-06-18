"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Link, ArrowRight, Zap } from "lucide-react";

export function MCPProtocolDemo() {
  const [step, setStep] = useState(0);
  const steps = ["Agent发送请求", "MCP路由分发", "工具执行",", 返回结果"];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Link className="w-6 h-6 text-violet-500" />
        MCP 协议流程
      </h3>

      <button onClick={() => setStep((s) => (s + 1) % 4)}
        className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 mb-6">下一步</button>

      <div className="flex items-center justify-between bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        {steps.map((s, idx) => (
          <div key={idx} className="flex items-center">
            <div className={`text-center ${idx === step ? "scale-110" : "opacity-50"}`}>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-2 ${idx === step ? "bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500" : "bg-slate-100 dark:bg-slate-800"}`}>
                <Zap className={`w-8 h-8 ${idx === step ? "text-violet-500" : "text-slate-400"}`} />
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{s}</span>
            </div>
            {idx < 3 && <ArrowRight className="w-6 h-6 text-slate-300 mx-2" />}
          </div>
        ))}
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code, Send, Play, Check } from "lucide-react";

const STEPS = [
  { id: 1, title: "定义函数", icon: Code, code: `def get_weather(city: str) -> dict:\n    """获取城市天气"""\n    return {"temp": 28, "condition": "晴"}` },
  { id: 2, title: "发送请求", icon: Send, code: `response = client.chat.completions.create(\n    model="gpt-4",\n    tools=[weather_tool],\n    messages=[{"role": "user", "content": "北京天气"}]\n)` },
  { id: 3, title: "执行调用", icon: Play, code: `# LLM 返回 tool_call\nresult = get_weather(city="北京")` },
  { id: 4, title: "返回结果", icon: Check, code: `# 将结果返回给 LLM 生成回答\n{"temp": 28, "condition": "晴"}` },
];

export function FunctionCallingSteps() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Function Calling 流程</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        Function Calling 让 LLM 能够调用外部函数，扩展其能力边界。
      </p>

      <div className="flex gap-2 mb-6">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          return (
            <button
              key={step.id}
              onClick={() => setCurrent(idx)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all ${
                current === idx
                  ? "bg-amber-600 text-white"
                  : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
              }`}
            >
              <Icon className="w-4 h-4" />
              {step.title}
            </button>
          );
        })}
      </div>

      <motion.div
        key={current}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
      >
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
          步骤 {STEPS[current].id}: {STEPS[current].title}
        </h4>
        <pre className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 text-sm text-slate-700 dark:text-slate-200 overflow-x-auto">
          {STEPS[current].code}
        </pre>
      </motion.div>
    </div>
  );
}

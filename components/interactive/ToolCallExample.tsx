"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code, Play, Check } from "lucide-react";

const EXAMPLES = [
  { name: "天气查询", tool: "get_weather", args: '{"city": "北京"}', result: '{"temp": 28, "condition": "晴"}' },
  { name: "数学计算", tool: "calculator", args: '{"expression": "2+2"}', result: '{"result": 4}' },
];

export function ToolCallExample() {
  const [selected, setSelected] = useState(0);
  const ex = EXAMPLES[selected];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">工具调用示例</h3>
      <div className="flex gap-2 mb-6">
        {EXAMPLES.map((e, i) => (
          <button key={i} onClick={() => setSelected(i)}
            className={`px-4 py-2 rounded-lg ${selected === i ? "bg-green-600 text-white" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"}`}>
            {e.name}
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
            <span className="text-xs text-green-600">工具</span>
            <code className="block text-sm text-green-700">{ex.tool}</code>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
            <span className="text-xs text-blue-600">参数</span>
            <code className="block text-sm text-blue-700">{ex.args}</code>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
            <span className="text-xs text-purple-600">结果</span>
            <code className="block text-sm text-purple-700">{ex.result}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

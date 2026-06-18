"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code, CheckCircle, Bug, Check } from "lucide-react";

const STEPS = [
  { id: 1, name: "理解需求", icon: Code, description: "分析用户需求" },
  { id: 2, name: "生成代码", icon: Code, description: "编写代码实现" },
  { id: 3, name: "运行测试", icon: CheckCircle, description: "执行单元测试" },
  { id: 4, name: "修复问题", icon: Bug, description: "修复测试失败" },
  { id: 5, name: "完成", icon: Check, description: "所有测试通过" },
];

export function CodeGenPipeline() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">代码生成 Agent 流程</h3>

      <button onClick={() => setCurrent((c) => (c + 1) % STEPS.length)}
        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 mb-6">
        下一步
      </button>

      <div className="flex items-center justify-between">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.id} className={`text-center ${idx === current ? "scale-110" : "opacity-50"}`}>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-2 ${idx === current ? "bg-green-100 dark:bg-green-900/30 border-2 border-green-500" : "bg-slate-100 dark:bg-slate-800"}`}>
                <Icon className={`w-8 h-8 ${idx === current ? "text-green-500" : "text-slate-400"}`} />
              </div>
              <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{step.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Shield, AlertTriangle, Check } from "lucide-react";

const CHECKS = [
  { id: 1, name: "提示注入检测", status: "pass", icon: Shield },
  { id: 2, name: "有害内容过滤", status: "warning", icon: AlertTriangle },
  { id: 3, name: "隐私数据保护", status: "pass", icon: Check },
];

export function SafetyAlignmentCheck() {
  const [current, setCurrent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">安全对齐检查</h3>

      <button onClick={() => setCurrent((c) => (c + 1) % CHECKS.length)}
        className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 mb-6">下一项</button>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          {React.createElement(CHECKS[current].icon, { className: `w-12 h-12 ${CHECKS[current].status === 'pass' ? 'text-green-500' : 'text-yellow-500'}` })}
          <div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{CHECKS[current].name}</span>
            <span className={`ml-2 px-2 py-1 rounded text-xs ${CHECKS[current].status === 'pass' ? 'bg-green-100 text-green-600' : 'bg-yellow-100 text-yellow-600'}`}>
              {CHECKS[current].status === 'pass' ? '通过' : '警告'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

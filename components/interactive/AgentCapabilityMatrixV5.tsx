"use client";

import React from "react";
import { Check, X } from "lucide-react";

const CAPS = [
  { name: "自然语言理解", traditional: true, agent: true },
  { name: "自主决策", traditional: false, agent: true },
  { name: "多轮交互", traditional: false, agent: true },
];

export function AgentCapabilityMatrixV5() {
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">能力矩阵V4</h3>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <table className="w-full text-sm">
          <thead><tr className="border-b"><th className="p-2 text-left">能力</th><th className="p-2 text-center">传统系统</th><th className="p-2 text-center">AI Agent</th></tr></thead>
          <tbody>
            {CAPS.map((c, i) => (
              <tr key={i} className="border-b">
                <td className="p-2 font-medium">{c.name}</td>
                <td className="text-center">{c.traditional ? <Check className="w-4 h-4 text-green-500 mx-auto"/> : <X className="w-4 h-4 text-red-400 mx-auto"/>}</td>
                <td className="text-center"><Check className="w-4 h-4 text-green-500 mx-auto"/></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

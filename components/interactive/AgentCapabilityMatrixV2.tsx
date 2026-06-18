"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Check, X } from "lucide-react";

const CAPABILITIES = [
  { feature: "自主决策", basic: false, advanced: true },
  { feature: "工具调用", basic: false, advanced: true },
  { feature: "记忆系统", basic: false, advanced: true },
];

export function AgentCapabilityMatrixV2() {
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent能力对比V2</h3>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <table className="w-full">
          <thead><tr className="border-b"><th className="p-2">功能</th><th className="p-2">基础模型</th><th className="p-2">Agent</th></tr></thead>
          <tbody>
            {CAPABILITIES.map((c, i) => (
              <tr key={i} className="border-b"><td className="p-2">{c.feature}</td><td className="text-center">{c.basic ? <Check className="w-5 h-5 text-green-500 mx-auto"/> : <X className="w-5 h-5 text-red-500 mx-auto"/>}</td><td className="text-center"><Check className="w-5 h-5 text-green-500 mx-auto"/></td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

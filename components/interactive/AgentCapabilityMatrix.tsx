"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Check, X } from "lucide-react";

const CAPABILITIES = [
  { name: "自主性", chatbot: false, agent: true },
  { name: "工具使用", chatbot: false, agent: true },
  { name: "记忆系统", chatbot: false, agent: true },
  { name: "规划能力", chatbot: false, agent: true },
];

export function AgentCapabilityMatrix() {
  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Chatbot vs Agent 能力对比</h3>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <table className="w-full">
          <thead>
            <tr className="border-b">
              <th className="text-left p-3 text-slate-600">能力</th>
              <th className="text-center p-3 text-slate-600">Chatbot</th>
              <th className="text-center p-3 text-slate-600">Agent</th>
            </tr>
          </thead>
          <tbody>
            {CAPABILITIES.map((cap, i) => (
              <tr key={i} className="border-b">
                <td className="p-3 font-medium text-slate-800 dark:text-slate-100">{cap.name}</td>
                <td className="text-center p-3">{cap.chatbot ? <Check className="w-5 h-5 text-green-500 mx-auto" /> : <X className="w-5 h-5 text-red-500 mx-auto" />}</td>
                <td className="text-center p-3"><Check className="w-5 h-5 text-green-500 mx-auto" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";

const LEVELS = [
  { id: "L1", name: "辅助型", description: "LLM仅提供建议", humanInLoop: "100%", example: "ChatGPT" },
  { id: "L2", name: "工具增强型", description: "可调用工具，需确认", humanInLoop: "高", example: "Copilot" },
  { id: "L3", name: "条件自主型", description: "限定范围自主执行", humanInLoop: "中", example: "客服Agent" },
  { id: "L4", name: "高度自主型", description: "复杂任务自主完成", humanInLoop: "低", example: "Devin" },
];

export function AgentAutonomyLevels() {
  const [selected, setSelected] = useState("L1");

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 自主性等级</h3>
      <div className="grid grid-cols-4 gap-3 mb-6">
        {LEVELS.map((l) => (
          <button key={l.id} onClick={() => setSelected(l.id)}
            className={`p-3 rounded-xl transition-all ${selected === l.id ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <span className="font-bold text-cyan-600">{l.id}</span>
            <span className="block text-sm text-slate-700 dark:text-slate-200">{l.name}</span>
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">{LEVELS.find(l => l.id === selected)?.name}</h4>
        <p className="text-slate-600 dark:text-slate-300">{LEVELS.find(l => l.id === selected)?.description}</p>
        <p className="text-sm text-slate-500 mt-2">人类参与度: {LEVELS.find(l => l.id === selected)?.humanInLoop}</p>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRightLeft, Server, Shield } from "lucide-react";

const PROTOCOLS = [
  { id: "mcp", name: "MCP", description: "Model Context Protocol", feature: "工具标准化" },
  { id: "a2a", name: "A2A", description: "Agent-to-Agent", feature: "Agent间通信" },
];

export function AgentProtocolDemo() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 通信协议</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">MCP 和 A2A 是两种主流的 Agent 通信协议。</p>

      <div className="flex gap-4 mb-6">
        {PROTOCOLS.map((p, idx) => (
          <button key={p.id} onClick={() => setSelected(idx)}
            className={`flex-1 p-4 rounded-xl transition-all ${selected === idx ? "bg-sky-100 dark:bg-sky-900/30 border-2 border-sky-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <ArrowRightLeft className={`w-8 h-8 mb-2 mx-auto ${selected === idx ? "text-sky-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{p.name}</span>
            <span className="text-xs text-slate-500">{p.description}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{PROTOCOLS[selected].name} 协议</h4>
        <p className="text-slate-600 dark:text-slate-300">核心特性: {PROTOCOLS[selected].feature}</p>
      </div>
    </div>
  );
}

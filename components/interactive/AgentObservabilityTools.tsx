"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Eye, Activity, Clock } from "lucide-react";

const TOOLS = [
  { id: 1, name: "LangSmith", icon: Eye, description: "追踪Agent执行轨迹" },
  { id: 2, name: "Prometheus", icon: Activity, description: "收集性能指标" },
  { id: 3, name: "Jaeger", icon: Clock, description: "分布式链路追踪" },
];

export function AgentObservabilityTools() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 可观测性工具</h3>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {TOOLS.map((t, idx) => (
          <button key={t.id} onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${selected === idx ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <t.icon className={`w-8 h-8 mb-2 ${selected === idx ? "text-cyan-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{t.name}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-300">{TOOLS[selected].description}</p>
      </div>
    </div>
  );
}

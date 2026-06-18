"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Activity, AlertCircle, CheckCircle } from "lucide-react";

const METRICS = [
  { id: 1, name: "请求延迟", value: "125ms", status: "good", icon: Activity },
  { id: 2, name: "错误率", value: "0.5%", status: "good", icon: CheckCircle },
  { id: 3, name: "Token 使用", value: "1.2M/天", status: "warning", icon: AlertCircle },
];

export function MonitoringDashboardDemo() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">监控仪表板</h3>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {METRICS.map((m, idx) => (
          <button key={m.id} onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${selected === idx ? "bg-indigo-100 dark:bg-indigo-900/30 border-2 border-indigo-500" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <m.icon className={`w-8 h-8 mb-2 ${selected === idx ? "text-indigo-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{m.name}</span>
            <span className={`text-2xl font-bold ${m.status === 'good' ? 'text-green-600' : 'text-yellow-600'}`}>{m.value}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

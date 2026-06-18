"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Server, Check, Clock } from "lucide-react";

const ITEMS = [
  { id: 1, name: "容器化部署", done: true, icon: Server },
  { id: 2, name: "负载均衡配置", done: true, icon: Check },
  { id: 3, name: "监控告警设置", done: false, icon: Clock },
];

export function ProductionDeploymentChecklist() {
  const [items, setItems] = useState(ITEMS);

  const toggleItem = (id: number) => {
    setItems(items.map(i => i.id === id ? { ...i, done: !i.done } : i));
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">生产部署检查清单</h3>

      <div className="space-y-3">
        {items.map(item => (
          <div key={item.id} onClick={() => toggleItem(item.id)}
            className={`flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all ${item.done ? "bg-teal-50 dark:bg-teal-900/20 border border-teal-200" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${item.done ? "bg-teal-500 text-white" : "bg-slate-200 dark:bg-slate-700"}`}>
              {item.done && <Check className="w-4 h-4" />}
            </div>
            <item.icon className={`w-5 h-5 ${item.done ? "text-teal-500" : "text-slate-400"}`} />
            <span className={`font-medium ${item.done ? "text-teal-700 dark:text-teal-300" : "text-slate-700 dark:text-slate-200"}`}>{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

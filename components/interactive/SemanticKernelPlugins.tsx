"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Plug, Settings, Zap } from "lucide-react";

interface Plugin {
  id: string;
  name: string;
  description: string;
  functions: string[];
}

const PLUGINS: Plugin[] = [
  { id: "weather", name: "天气插件", description: "获取天气信息", functions: ["get_current_weather", "get_forecast"] },
  { id: "calendar", name: "日历插件", description: "管理日程", functions: ["create_event", "list_events"] },
  { id: "email", name: "邮件插件", description: "发送邮件", functions: ["send_email", "read_inbox"] },
];

export function SemanticKernelPlugins() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Plug className="w-6 h-6 text-blue-500" />
        Semantic Kernel 插件系统
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        插件让Agent能够调用外部服务和API。
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {PLUGINS.map((plugin, idx) => (
          <button
            key={plugin.id}
            onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${
              selected === idx
                ? "bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500"
                : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
            }`}
          >
            <Plug className={`w-6 h-6 mb-2 ${selected === idx ? "text-blue-500" : "text-slate-400"}`} />
            <span className="font-bold text-slate-800 dark:text-slate-100 block">{plugin.name}</span>
            <span className="text-xs text-slate-500">{plugin.description}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{PLUGINS[selected].name} 函数</h4>
        <div className="space-y-2">
          {PLUGINS[selected].functions.map((fn, i) => (
            <div key={i} className="flex items-center gap-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
              <Zap className="w-4 h-4 text-blue-500" />
              <code className="text-sm text-blue-700 dark:text-blue-300">{fn}</code>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

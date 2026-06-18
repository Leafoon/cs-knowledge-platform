"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Database, Clock, Brain } from "lucide-react";

const MEMORY_TYPES = [
  { id: "short", name: "短期记忆", icon: Clock, color: "blue", description: "当前对话上下文，容量有限", example: "对话历史、临时状态" },
  { id: "long", name: "长期记忆", icon: Database, color: "green", description: "持久化的重要信息", example: "用户偏好、历史知识" },
  { id: "working", name: "工作记忆", icon: Brain, color: "purple", description: "当前任务的中间状态", example: "推理步骤、工具结果" },
];

export function MemoryArchitecture() {
  const [selected, setSelected] = useState("short");

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">Agent 记忆架构</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        记忆系统让 Agent 能够记住历史、学习偏好、积累知识。
      </p>

      <div className="flex gap-4 mb-6">
        {MEMORY_TYPES.map((type) => {
          const Icon = type.icon;
          return (
            <button
              key={type.id}
              onClick={() => setSelected(type.id)}
              className={`flex items-center gap-2 px-4 py-3 rounded-xl transition-all ${
                selected === type.id
                  ? `bg-${type.color}-100 dark:bg-${type.color}-900/30 border-2 border-${type.color}-500`
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
              }`}
            >
              <Icon className={`w-5 h-5 ${selected === type.id ? `text-${type.color}-500` : "text-slate-400"}`} />
              <span className="font-medium text-slate-700 dark:text-slate-200">{type.name}</span>
            </button>
          );
        })}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
          {MEMORY_TYPES.find((t) => t.id === selected)?.name}
        </h4>
        <p className="text-slate-600 dark:text-slate-300 mb-3">
          {MEMORY_TYPES.find((t) => t.id === selected)?.description}
        </p>
        <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
          <span className="text-sm text-slate-500">典型数据: </span>
          <span className="text-sm text-slate-700 dark:text-slate-200">
            {MEMORY_TYPES.find((t) => t.id === selected)?.example}
          </span>
        </div>
      </div>
    </div>
  );
}

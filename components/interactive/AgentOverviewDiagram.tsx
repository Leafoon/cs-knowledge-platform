"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Brain, Wrench, Database, Target } from "lucide-react";

const COMPONENTS = [
  { id: "llm", name: "LLM 基座", icon: Brain, color: "blue", description: "推理与理解能力" },
  { id: "tools", name: "工具集", icon: Wrench, color: "green", description: "与外部世界交互" },
  { id: "memory", name: "记忆系统", icon: Database, color: "purple", description: "存储历史信息" },
  { id: "planning", name: "规划模块", icon: Target, color: "orange", description: "任务分解与规划" },
];

export function AgentOverviewDiagram() {
  const [active, setActive] = useState<string | null>(null);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">Agent 核心架构</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        一个完整的 AI Agent 由四大核心组件构成，它们协同工作实现智能行为。
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {COMPONENTS.map((comp) => {
          const Icon = comp.icon;
          const isActive = active === comp.id;
          return (
            <motion.div
              key={comp.id}
              onClick={() => setActive(isActive ? null : comp.id)}
              className={`p-6 rounded-xl cursor-pointer text-center transition-all ${
                isActive
                  ? `bg-${comp.color}-100 dark:bg-${comp.color}-900/30 border-2 border-${comp.color}-500`
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-300"
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Icon className={`w-12 h-12 mx-auto mb-3 ${isActive ? `text-${comp.color}-500` : "text-slate-400"}`} />
              <h4 className="font-bold text-slate-800 dark:text-slate-100">{comp.name}</h4>
              <p className="text-sm text-slate-500 mt-1">{comp.description}</p>
            </motion.div>
          );
        })}
      </div>

      {active && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700"
        >
          <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-3">
            {COMPONENTS.find((c) => c.id === active)?.name}
          </h4>
          <p className="text-slate-600 dark:text-slate-300">
            {active === "llm" && "大语言模型是 Agent 的推理核心，负责理解自然语言、进行逻辑推理、生成回答和决策。不同的 LLM 在能力、成本、延迟上有显著差异。"}
            {active === "tools" && "工具集赋予 Agent 与外部世界交互的能力，包括 API 调用、数据库查询、代码执行、文件操作等。工具设计遵循单一职责原则。"}
            {active === "memory" && "记忆系统让 Agent 能够记住历史对话、学习用户偏好、积累知识。包括短期记忆（当前对话）和长期记忆（跨会话）。"}
            {active === "planning" && "规划模块使 Agent 能够分解复杂任务、制定执行计划、动态调整策略。支持 Chain-of-Thought、Tree of Thoughts 等算法。"}
          </p>
        </motion.div>
      )}
    </div>
  );
}

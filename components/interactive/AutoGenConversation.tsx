"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Users, MessageCircle } from "lucide-react";

interface Agent {
  id: string;
  name: string;
  role: string;
  color: string;
}

const AGENTS: Agent[] = [
  { id: "planner", name: "规划师", role: "制定计划", color: "blue" },
  { id: "coder", name: "程序员", role: "编写代码", color: "green" },
  { id: "reviewer", name: "审查员", role: "审查代码", color: "purple" },
];

export function AutoGenConversation() {
  const [activeAgent, setActiveAgent] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Users className="w-6 h-6 text-cyan-500" />
        AutoGen 多Agent对话
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        多个Agent通过对话协作完成复杂任务。
      </p>

      <div className="flex gap-4 mb-6">
        {AGENTS.map((agent, idx) => (
          <button
            key={agent.id}
            onClick={() => setActiveAgent(idx)}
            className={`flex-1 p-4 rounded-xl transition-all ${
              activeAgent === idx
                ? `bg-${agent.color}-100 dark:bg-${agent.color}-900/30 border-2 border-${agent.color}-500`
                : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
            }`}
          >
            <span className={`font-bold ${activeAgent === idx ? `text-${agent.color}-600` : "text-slate-600"}`}>
              {agent.name}
            </span>
            <span className="block text-xs text-slate-500">{agent.role}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-start gap-3">
          <MessageCircle className="w-6 h-6 text-cyan-500 mt-1" />
          <div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{AGENTS[activeAgent].name}</span>
            <p className="text-slate-600 dark:text-slate-300 mt-1">
              {activeAgent === 0 && "我来规划这个任务的执行步骤..."}
              {activeAgent === 1 && "根据规划，我来编写代码实现..."}
              {activeAgent === 2 && "我来审查代码质量和安全性..."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

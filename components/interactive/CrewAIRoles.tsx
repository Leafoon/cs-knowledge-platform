"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { User, Target, BookOpen } from "lucide-react";

interface Role {
  id: string;
  name: string;
  goal: string;
  backstory: string;
  icon: React.ReactNode;
}

const ROLES: Role[] = [
  { id: "researcher", name: "研究员", goal: "收集和分析信息", backstory: "10年研究经验", icon: <BookOpen className="w-6 h-6" /> },
  { id: "analyst", name: "分析师", goal: "数据分析和洞察", backstory: "数据科学专家", icon: <Target className="w-6 h-6" /> },
  { id: "writer", name: "写手", goal: "撰写报告和文档", backstory: "资深编辑", icon: <User className="w-6 h-6" /> },
];

export function CrewAIRoles() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">CrewAI 角色定义</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        每个Agent有明确的角色、目标和背景故事。
      </p>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {ROLES.map((role, idx) => (
          <button
            key={role.id}
            onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl transition-all ${
              selected === idx
                ? "bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500"
                : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
            }`}
          >
            <div className={`${selected === idx ? "text-orange-500" : "text-slate-400"} mb-2`}>{role.icon}</div>
            <span className="font-bold text-slate-800 dark:text-slate-100">{role.name}</span>
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{ROLES[selected].name}</h4>
        <div className="space-y-2">
          <div><span className="text-sm text-slate-500">目标: </span><span className="text-slate-700 dark:text-slate-200">{ROLES[selected].goal}</span></div>
          <div><span className="text-sm text-slate-500">背景: </span><span className="text-slate-700 dark:text-slate-200">{ROLES[selected].backstory}</span></div>
        </div>
      </div>
    </div>
  );
}

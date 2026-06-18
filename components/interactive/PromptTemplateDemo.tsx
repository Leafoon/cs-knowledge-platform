"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Edit3, Eye } from "lucide-react";

const TEMPLATES = [
  { id: "zero", name: "Zero-Shot", prompt: "将以下文本分类为正面/负面情感：\n{text}", examples: 0 },
  { id: "few", name: "Few-Shot", prompt: "示例：'太棒了' → 正面\n示例：'很差劲' → 负面\n\n将以下文本分类：\n{text}", examples: 2 },
  { id: "cot", name: "Chain-of-Thought", prompt: "请一步步思考：\n1. 首先分析文本的情感词\n2. 判断整体情感倾向\n3. 给出分类结果\n\n文本：{text}", examples: 0 },
];

export function PromptTemplateDemo() {
  const [selected, setSelected] = useState("zero");
  const template = TEMPLATES.find((t) => t.id === selected)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <FileText className="w-6 h-6 text-violet-500" />
        Prompt 工程模式对比
      </h3>

      <div className="flex gap-3 mb-6">
        {TEMPLATES.map((t) => (
          <button
            key={t.id}
            onClick={() => setSelected(t.id)}
            className={`px-4 py-2 rounded-lg transition-all ${
              selected === t.id
                ? "bg-violet-600 text-white"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-3">
          <Edit3 className="w-5 h-5 text-violet-500" />
          <span className="font-bold text-slate-800 dark:text-slate-100">{template.name} 模板</span>
        </div>
        <pre className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 text-sm text-slate-700 dark:text-slate-200 overflow-x-auto">
          {template.prompt}
        </pre>
        <div className="mt-3 text-sm text-slate-500">
          示例数量: {template.examples}
        </div>
      </div>
    </div>
  );
}

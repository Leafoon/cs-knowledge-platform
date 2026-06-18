"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Settings, Sliders, ArrowRight, Check } from "lucide-react";

interface TuningParameter {
  id: string;
  name: string;
  description: string;
  defaultValue: string;
  recommendedRange: string;
  impact: "high" | "medium" | "low";
  tips: string[];
}

const PARAMETERS: TuningParameter[] = [
  {
    id: "chunk_size",
    name: "分块大小 (chunk_size)",
    description: "每个文本块的字符数或token数",
    defaultValue: "1000",
    recommendedRange: "500-2000",
    impact: "high",
    tips: ["小块提高精度，大块保持上下文", "代码块建议2000+", "对话建议500-800"],
  },
  {
    id: "chunk_overlap",
    name: "分块重叠 (chunk_overlap)",
    description: "相邻块之间的重叠字符数",
    defaultValue: "200",
    recommendedRange: "100-400",
    impact: "medium",
    tips: ["重叠太多增加存储成本", "太少可能切断语义", "建议为chunk_size的10-20%"],
  },
  {
    id: "top_k",
    name: "检索数量 (top_k)",
    description: "返回最相关的k个文档块",
    defaultValue: "4",
    recommendedRange: "3-10",
    impact: "high",
    tips: ["太多引入噪声，太少遗漏信息", "复杂问题用更大k", "配合重排序效果更好"],
  },
  {
    id: "temperature",
    name: "生成温度 (temperature)",
    description: "控制LLM生成的随机性",
    defaultValue: "0.7",
    recommendedRange: "0.3-1.0",
    impact: "medium",
    tips: ["事实性问题用低温", "创意任务用高温", "RAG推荐0.3-0.5"],
  },
];

export function RAGTuningGuide() {
  const [selectedParam, setSelectedParam] = useState<string>("chunk_size");
  const [userValues, setUserValues] = useState<Record<string, string>>({});
  const param = PARAMETERS.find((p) => p.id === selectedParam)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Settings className="w-6 h-6 text-amber-500" />
        RAG 参数调优指南
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 参数列表 */}
        <div className="space-y-3">
          {PARAMETERS.map((p) => (
            <motion.button
              key={p.id}
              onClick={() => setSelectedParam(p.id)}
              className={`w-full p-4 rounded-xl text-left transition-all ${
                selectedParam === p.id
                  ? "bg-amber-100 dark:bg-amber-900/30 border-2 border-amber-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-amber-300"
              }`}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex justify-between items-center">
                <span className="font-medium text-slate-700 dark:text-slate-200">{p.name}</span>
                <span className={`text-xs px-2 py-1 rounded ${
                  p.impact === "high" ? "bg-red-100 text-red-600" :
                  p.impact === "medium" ? "bg-yellow-100 text-yellow-600" :
                  "bg-green-100 text-green-600"
                }`}>
                  影响: {p.impact === "high" ? "高" : p.impact === "medium" ? "中" : "低"}
                </span>
              </div>
            </motion.button>
          ))}
        </div>

        {/* 参数详情 */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">{param.name}</h4>
          <p className="text-slate-600 dark:text-slate-300 mb-4">{param.description}</p>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
              <span className="text-sm text-slate-500">默认值</span>
              <p className="font-mono font-medium text-slate-700 dark:text-slate-200">{param.defaultValue}</p>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
              <span className="text-sm text-slate-500">推荐范围</span>
              <p className="font-mono font-medium text-amber-600 dark:text-amber-400">{param.recommendedRange}</p>
            </div>
          </div>

          {/* 输入框 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-2">
              设置你的值:
            </label>
            <input
              type="text"
              value={userValues[param.id] || ""}
              onChange={(e) => setUserValues({ ...userValues, [param.id]: e.target.value })}
              placeholder={param.defaultValue}
              className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200"
            />
          </div>

          <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">调优建议</h5>
          <ul className="space-y-2">
            {param.tips.map((tip, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-300">
                <Check className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                {tip}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

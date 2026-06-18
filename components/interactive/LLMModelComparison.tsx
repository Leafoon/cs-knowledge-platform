"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, Zap, DollarSign, Clock } from "lucide-react";

interface LLMModel {
  name: string;
  provider: string;
  params: string;
  context: string;
  latency: string;
  cost: string;
  bestFor: string;
}

const MODELS: LLMModel[] = [
  { name: "GPT-4o", provider: "OpenAI", params: "~200B", context: "128K", latency: "中", cost: "高", bestFor: "复杂推理、工具调用" },
  { name: "Claude 4", provider: "Anthropic", params: "未公开", context: "200K", latency: "中", cost: "高", bestFor: "长文本、代码生成" },
  { name: "GPT-4o-mini", provider: "OpenAI", params: "未公开", context: "128K", latency: "低", cost: "低", bestFor: "简单任务、快速响应" },
  { name: "DeepSeek V3", provider: "DeepSeek", params: "671B MoE", context: "128K", latency: "低", cost: "低", bestFor: "性价比、中文场景" },
];

export function LLMModelComparison() {
  const [selected, setSelected] = useState<number>(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <Cpu className="w-6 h-6 text-cyan-500" />
        主流 LLM 模型对比
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        选择合适的 LLM 是构建 Agent 的关键决策。不同模型在能力、成本、延迟上有显著差异。
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {MODELS.map((model, idx) => (
          <motion.div
            key={model.name}
            onClick={() => setSelected(idx)}
            className={`p-4 rounded-xl cursor-pointer transition-all ${
              selected === idx
                ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500"
                : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-cyan-300"
            }`}
            whileHover={{ scale: 1.02 }}
          >
            <h4 className="font-bold text-slate-800 dark:text-slate-100">{model.name}</h4>
            <p className="text-sm text-slate-500">{model.provider}</p>
          </motion.div>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">{MODELS[selected].name} 详情</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <Cpu className="w-5 h-5 text-cyan-500 mb-1" />
            <span className="text-xs text-slate-500 block">参数规模</span>
            <span className="font-medium text-slate-700 dark:text-slate-200">{MODELS[selected].params}</span>
          </div>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <Zap className="w-5 h-5 text-yellow-500 mb-1" />
            <span className="text-xs text-slate-500 block">上下文窗口</span>
            <span className="font-medium text-slate-700 dark:text-slate-200">{MODELS[selected].context}</span>
          </div>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <Clock className="w-5 h-5 text-green-500 mb-1" />
            <span className="text-xs text-slate-500 block">延迟</span>
            <span className="font-medium text-slate-700 dark:text-slate-200">{MODELS[selected].latency}</span>
          </div>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <DollarSign className="w-5 h-5 text-red-500 mb-1" />
            <span className="text-xs text-slate-500 block">成本</span>
            <span className="font-medium text-slate-700 dark:text-slate-200">{MODELS[selected].cost}</span>
          </div>
        </div>
        <div className="mt-4 p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg">
          <span className="text-sm font-medium text-cyan-700 dark:text-cyan-300">最佳场景: </span>
          <span className="text-cyan-600 dark:text-cyan-200">{MODELS[selected].bestFor}</span>
        </div>
      </div>
    </div>
  );
}

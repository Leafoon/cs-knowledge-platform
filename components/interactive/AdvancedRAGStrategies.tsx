"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Zap, Brain, GitBranch, ArrowRight } from "lucide-react";

interface RAGStrategy {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  keyIdea: string;
  pros: string[];
  cons: string[];
}

const STRATEGIES: RAGStrategy[] = [
  {
    id: "multi-query",
    name: "Multi-Query",
    icon: <Layers className="w-5 h-5" />,
    description: "将单一查询扩展为多个不同角度的查询，分别检索后合并结果。",
    keyIdea: "一个问题可以从多个角度提问",
    pros: ["提高召回率", "覆盖更多信息"],
    cons: ["增加延迟", "需要LLM生成查询"],
  },
  {
    id: "hyde",
    name: "HyDE",
    icon: <Brain className="w-5 h-5" />,
    description: "假设性文档嵌入：先让LLM生成一个假设的答案，用这个答案进行检索。",
    keyIdea: "假设的答案与真实文档更接近",
    pros: ["提高语义匹配", "适合开放性问题"],
    cons: ["依赖LLM质量", "可能引入偏差"],
  },
  {
    id: "self-rag",
    name: "Self-RAG",
    icon: <GitBranch className="w-5 h-5" />,
    description: "让LLM自我评估检索结果的相关性，决定是否需要更多信息。",
    keyIdea: "不是所有问题都需要检索",
    pros: ["智能路由", "减少不必要的检索"],
    cons: ["需要训练特殊token", "实现复杂"],
  },
  {
    id: "graph-rag",
    name: "Graph RAG",
    icon: <Zap className="w-5 h-5" />,
    description: "使用知识图谱增强检索，捕获实体之间的关系。",
    keyIdea: "关系和实体同样重要",
    pros: ["捕获关系", "支持多跳推理"],
    cons: ["图谱构建成本高", "维护复杂"],
  },
];

export function AdvancedRAGStrategies() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>("multi-query");
  const strategy = STRATEGIES.find((s) => s.id === selectedStrategy)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Layers className="w-6 h-6 text-orange-500" />
        高级 RAG 策略
      </h3>

      {/* 策略选择器 */}
      <div className="flex flex-wrap gap-3 mb-6">
        {STRATEGIES.map((s) => (
          <motion.button
            key={s.id}
            onClick={() => setSelectedStrategy(s.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              selectedStrategy === s.id
                ? "bg-orange-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-orange-100"
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {s.icon}
            {s.name}
          </motion.button>
        ))}
      </div>

      {/* 策略详情 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={strategy.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
        >
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
            {strategy.icon}
            {strategy.name}
          </h4>
          <p className="text-slate-600 dark:text-slate-300 mb-4">{strategy.description}</p>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 mb-4">
            <span className="text-sm font-medium text-orange-700 dark:text-orange-300">核心思想: </span>
            <span className="text-orange-600 dark:text-orange-200">{strategy.keyIdea}</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">优点</h5>
              <ul className="space-y-1">
                {strategy.pros.map((p, i) => (
                  <li key={i} className="text-sm text-slate-600 dark:text-slate-300">✓ {p}</li>
                ))}
              </ul>
            </div>
            <div>
              <h5 className="font-semibold text-red-600 dark:text-red-400 mb-2">缺点</h5>
              <ul className="space-y-1">
                {strategy.cons.map((c, i) => (
                  <li key={i} className="text-sm text-slate-600 dark:text-slate-300">✗ {c}</li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

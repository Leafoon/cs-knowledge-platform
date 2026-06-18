"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Database, Search, Zap, Clock, ArrowRight } from "lucide-react";

interface RetrievalStrategy {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  useCases: string[];
  pros: string[];
  cons: string[];
  latency: string;
  accuracy: string;
}

const STRATEGIES: RetrievalStrategy[] = [
  {
    id: "keyword",
    name: "关键词检索",
    icon: <Search className="w-5 h-5" />,
    description: "基于关键词精确匹配的检索方式，使用BM25等算法计算文档与查询的相关性。",
    useCases: ["精确查找", "代码搜索", "日志检索"],
    pros: ["速度快", "精确匹配", "可解释性强"],
    cons: ["无法理解语义", "同义词问题"],
    latency: "低 (<10ms)",
    accuracy: "中等",
  },
  {
    id: "semantic",
    name: "语义检索",
    icon: <Zap className="w-5 h-5" />,
    description: "使用向量嵌入将文本转换为稠密向量，通过余弦相似度进行检索。",
    useCases: ["问答系统", "文档推荐", "语义搜索"],
    pros: ["理解语义", "同义词匹配", "泛化能力强"],
    cons: ["需要向量模型", "计算成本高"],
    latency: "中 (50-200ms)",
    accuracy: "高",
  },
  {
    id: "hybrid",
    name: "混合检索",
    icon: <Database className="w-5 h-5" />,
    description: "结合关键词检索和语义检索的优势，使用RRF等算法融合排序结果。",
    useCases: ["生产环境", "高质量问答", "复杂查询"],
    pros: ["兼顾精确和语义", "鲁棒性强"],
    cons: ["实现复杂", "需要调参"],
    latency: "中 (100-300ms)",
    accuracy: "最高",
  },
  {
    id: "temporal",
    name: "时序检索",
    icon: <Clock className="w-5 h-5" />,
    description: "考虑时间因素的检索策略，新近的信息权重更高。",
    useCases: ["新闻检索", "对话历史", "实时数据"],
    pros: ["时效性强", "适合动态数据"],
    cons: ["可能忽略旧的重要信息"],
    latency: "低 (<50ms)",
    accuracy: "中等",
  },
];

export function MemoryRetrievalStrategies() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>("hybrid");
  const strategy = STRATEGIES.find((s) => s.id === selectedStrategy)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Database className="w-6 h-6 text-indigo-500" />
        记忆检索策略对比
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 策略选择列表 */}
        <div className="space-y-3">
          {STRATEGIES.map((s) => (
            <motion.button
              key={s.id}
              onClick={() => setSelectedStrategy(s.id)}
              className={`w-full p-4 rounded-xl text-left transition-all ${
                selectedStrategy === s.id
                  ? "bg-indigo-100 dark:bg-indigo-900/30 border-2 border-indigo-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3">
                <span className={`${selectedStrategy === s.id ? "text-indigo-600" : "text-slate-400"}`}>
                  {s.icon}
                </span>
                <span className="font-semibold text-slate-700 dark:text-slate-200">{s.name}</span>
              </div>
            </motion.button>
          ))}
        </div>

        {/* 策略详情 */}
        <div className="lg:col-span-2">
          <AnimatePresence mode="wait">
            <motion.div
              key={strategy.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
            >
              <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
                {strategy.icon}
                {strategy.name}
              </h4>
              <p className="text-slate-600 dark:text-slate-300 mb-4">{strategy.description}</p>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
                  <span className="text-sm text-slate-500">延迟</span>
                  <p className="font-medium text-slate-700 dark:text-slate-200">{strategy.latency}</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
                  <span className="text-sm text-slate-500">准确率</span>
                  <p className="font-medium text-slate-700 dark:text-slate-200">{strategy.accuracy}</p>
                </div>
              </div>

              <div className="mb-4">
                <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">适用场景</h5>
                <div className="flex flex-wrap gap-2">
                  {strategy.useCases.map((uc, i) => (
                    <span key={i} className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded text-sm">
                      {uc}
                    </span>
                  ))}
                </div>
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
      </div>
    </div>
  );
}

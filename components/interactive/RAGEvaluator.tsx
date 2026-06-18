"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Check, X, AlertTriangle, BarChart3 } from "lucide-react";

interface EvaluationMetric {
  name: string;
  score: number;
  description: string;
  threshold: number;
}

const METRICS: EvaluationMetric[] = [
  { name: "Faithfulness", score: 0.92, description: "回答是否忠实于检索到的上下文", threshold: 0.8 },
  { name: "Answer Relevance", score: 0.88, description: "回答与问题的相关性", threshold: 0.75 },
  { name: "Context Precision", score: 0.85, description: "检索到的上下文的精确度", threshold: 0.7 },
  { name: "Context Recall", score: 0.79, description: "检索到的上下文的召回率", threshold: 0.75 },
  { name: "Retrieval Accuracy", score: 0.91, description: "检索结果的准确率", threshold: 0.85 },
];

export function RAGEvaluator() {
  const [selectedMetric, setSelectedMetric] = useState<number | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <BarChart3 className="w-6 h-6 text-blue-500" />
        RAG 系统评估指标
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 左侧：指标列表 */}
        <div className="space-y-3">
          {METRICS.map((metric, idx) => (
            <motion.div
              key={metric.name}
              className={`p-4 rounded-xl cursor-pointer transition-all ${
                selectedMetric === idx
                  ? "bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-300"
              }`}
              onClick={() => setSelectedMetric(idx)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold text-slate-700 dark:text-slate-200">
                  {metric.name}
                </span>
                <span className={`px-2 py-1 rounded text-sm font-medium ${
                  metric.score >= metric.threshold
                    ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                    : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                }`}>
                  {(metric.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <motion.div
                  className={`h-2 rounded-full ${
                    metric.score >= metric.threshold ? "bg-green-500" : "bg-yellow-500"
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${metric.score * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.1 }}
                />
              </div>
            </motion.div>
          ))}
        </div>

        {/* 右侧：详情面板 */}
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          {selectedMetric !== null ? (
            <div>
              <h4 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">
                {METRICS[selectedMetric].name}
              </h4>
              <p className="text-slate-600 dark:text-slate-300 mb-4">
                {METRICS[selectedMetric].description}
              </p>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-500">当前得分:</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">
                    {(METRICS[selectedMetric].score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">及格线:</span>
                  <span className="font-bold text-slate-700 dark:text-slate-300">
                    {(METRICS[selectedMetric].threshold * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">状态:</span>
                  {METRICS[selectedMetric].score >= METRICS[selectedMetric].threshold ? (
                    <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                      <Check className="w-4 h-4" /> 通过
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-yellow-600 dark:text-yellow-400">
                      <AlertTriangle className="w-4 h-4" /> 需改进
                    </span>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-400 dark:text-slate-500 py-12">
              ← 点击左侧指标查看详情
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

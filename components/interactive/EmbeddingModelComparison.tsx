"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Zap, Globe, DollarSign } from "lucide-react";

interface EmbeddingModel {
  id: string;
  name: string;
  provider: string;
  dimensions: number;
  performance: number;
  cost: number;
  languages: string[];
  description: string;
}

const MODELS: EmbeddingModel[] = [
  {
    id: "text-embedding-3-small",
    name: "text-embedding-3-small",
    provider: "OpenAI",
    dimensions: 1536,
    performance: 85,
    cost: 0.02,
    languages: ["英文", "中文", "多语言"],
    description: "性价比最高的选择，适合大多数场景",
  },
  {
    id: "text-embedding-3-large",
    name: "text-embedding-3-large",
    provider: "OpenAI",
    dimensions: 3072,
    performance: 95,
    cost: 0.13,
    languages: ["英文", "中文", "多语言"],
    description: "最高性能，适合对质量要求高的场景",
  },
  {
    id: "bge-large-zh",
    name: "bge-large-zh-v1.5",
    provider: "BAAI",
    dimensions: 1024,
    performance: 88,
    cost: 0,
    languages: ["中文", "英文"],
    description: "中文场景最佳选择，开源免费",
  },
  {
    id: "e5-large-v2",
    name: "e5-large-v2",
    provider: "Microsoft",
    dimensions: 1024,
    performance: 82,
    cost: 0,
    languages: ["英文", "多语言"],
    description: "微软开源模型，性能稳定",
  },
];

export function EmbeddingModelComparison() {
  const [selectedModel, setSelectedModel] = useState<string>("text-embedding-3-small");
  const [sortBy, setSortBy] = useState<"performance" | "cost">("performance");

  const sortedModels = [...MODELS].sort((a, b) =>
    sortBy === "performance" ? b.performance - a.performance : a.cost - b.cost
  );

  const selected = MODELS.find((m) => m.id === selectedModel)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Layers className="w-6 h-6 text-violet-500" />
        Embedding 模型对比
      </h3>

      {/* 排序控制 */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSortBy("performance")}
          className={`px-4 py-2 rounded-lg transition-all ${
            sortBy === "performance"
              ? "bg-violet-600 text-white"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
          }`}
        >
          <Zap className="w-4 h-4 inline mr-1" /> 按性能排序
        </button>
        <button
          onClick={() => setSortBy("cost")}
          className={`px-4 py-2 rounded-lg transition-all ${
            sortBy === "cost"
              ? "bg-violet-600 text-white"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"
          }`}
        >
          <DollarSign className="w-4 h-4 inline mr-1" /> 按成本排序
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 模型列表 */}
        <div className="space-y-3">
          {sortedModels.map((model) => (
            <motion.div
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`p-4 rounded-xl cursor-pointer transition-all ${
                selectedModel === model.id
                  ? "bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-violet-300"
              }`}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <span className="font-semibold text-slate-700 dark:text-slate-200">{model.name}</span>
                  <span className="ml-2 text-sm text-slate-500">{model.provider}</span>
                </div>
                <span className="text-sm font-medium text-violet-600 dark:text-violet-400">
                  {model.performance}%
                </span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <motion.div
                  className="h-2 bg-violet-500 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${model.performance}%` }}
                />
              </div>
            </motion.div>
          ))}
        </div>

        {/* 详情面板 */}
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">{selected.name}</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-500">提供商:</span>
              <span className="font-medium text-slate-700 dark:text-slate-200">{selected.provider}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">维度:</span>
              <span className="font-medium text-slate-700 dark:text-slate-200">{selected.dimensions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">成本:</span>
              <span className="font-medium text-slate-700 dark:text-slate-200">
                {selected.cost === 0 ? "免费" : `$${selected.cost}/1M tokens`}
              </span>
            </div>
            <div>
              <span className="text-slate-500">支持语言:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {selected.languages.map((lang, i) => (
                  <span key={i} className="px-2 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 rounded text-sm">
                    {lang}
                  </span>
                ))}
              </div>
            </div>
            <p className="text-slate-600 dark:text-slate-300 mt-4 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
              {selected.description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

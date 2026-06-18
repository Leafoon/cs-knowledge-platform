"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowUpDown, Check } from "lucide-react";

interface Doc {
  id: number;
  content: string;
  initialRank: number;
  finalRank: number;
  score: number;
}

const DOCS: Doc[] = [
  { id: 1, content: "机器学习是人工智能的分支", initialRank: 3, finalRank: 1, score: 0.95 },
  { id: 2, content: "深度学习使用神经网络", initialRank: 1, finalRank: 2, score: 0.88 },
  { id: 3, content: "数据预处理很重要", initialRank: 2, finalRank: 3, score: 0.72 },
];

export function ReRankingDemo() {
  const [reranked, setReranked] = useState(false);
  const sorted = reranked ? [...DOCS].sort((a, b) => a.finalRank - b.finalRank) : [...DOCS].sort((a, b) => a.initialRank - b.initialRank);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
        <ArrowUpDown className="w-6 h-6 text-indigo-500" />
        重排序演示
      </h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        重排序对初始检索结果进行精排，提高最终结果的相关性。
      </p>

      <button
        onClick={() => setReranked(!reranked)}
        className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 mb-6"
      >
        {reranked ? "显示初始排序" : "执行重排序"}
      </button>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-4">
          {reranked ? "重排序后" : "初始检索结果"}
        </h4>
        <div className="space-y-3">
          {sorted.map((doc, idx) => (
            <motion.div
              key={doc.id}
              layout
              className={`p-4 rounded-lg border ${
                reranked && idx === 0
                  ? "bg-green-50 dark:bg-green-900/20 border-green-300"
                  : "bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700"
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center text-sm font-bold text-indigo-600">
                  #{idx + 1}
                </span>
                <span className="flex-1 text-slate-700 dark:text-slate-200">{doc.content}</span>
                {reranked && <Check className="w-5 h-5 text-green-500" />}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

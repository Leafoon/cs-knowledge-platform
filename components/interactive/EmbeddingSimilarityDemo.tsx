"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Check } from "lucide-react";

const SENTENCES = [
  { id: 1, text: "今天天气真好" },
  { id: 2, text: "今天阳光明媚" },
  { id: 3, text: "我喜欢吃苹果" },
];

export function EmbeddingSimilarityDemo() {
  const [query, setQuery] = useState(0);
  const similarities = [
    [1.0, 0.85, 0.12],
    [0.85, 1.0, 0.15],
    [0.12, 0.15, 1.0],
  ];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">语义相似度演示</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        Embedding 将文本转换为向量，通过余弦相似度计算语义相关性。
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">选择查询文本</h4>
          <div className="space-y-2">
            {SENTENCES.map((s, idx) => (
              <button
                key={s.id}
                onClick={() => setQuery(idx)}
                className={`w-full p-3 rounded-lg text-left transition-all ${
                  query === idx
                    ? "bg-violet-100 dark:bg-violet-900/30 border-2 border-violet-500"
                    : "bg-slate-50 dark:bg-slate-900 hover:bg-slate-100"
                }`}
              >
                <span className="text-slate-700 dark:text-slate-200">{s.text}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">相似度结果</h4>
          <div className="space-y-3">
            {SENTENCES.map((s, idx) => (
              <div key={s.id}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-600 dark:text-slate-300">{s.text}</span>
                  <span className="font-medium text-violet-600 dark:text-violet-400">
                    {(similarities[query][idx] * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <motion.div
                    className="h-2 bg-violet-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${similarities[query][idx] * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

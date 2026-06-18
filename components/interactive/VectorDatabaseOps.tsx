"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, Plus, Trash2 } from "lucide-react";

interface VectorItem {
  id: number;
  text: string;
  similarity?: number;
}

const SAMPLE_DB: VectorItem[] = [
  { id: 1, text: "Python是一种编程语言" },
  { id: 2, text: "机器学习需要大量数据" },
  { id: 3, text: "向量数据库存储嵌入向量" },
];

export function VectorDatabaseOps() {
  const [db, setDb] = useState<VectorItem[]>(SAMPLE_DB);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<VectorItem[]>([]);

  const handleSearch = () => {
    if (!query) return;
    const scored = db.map((item) => ({
      ...item,
      similarity: Math.random() * 0.5 + 0.5,
    }));
    scored.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    setResults(scored.slice(0, 2));
  };

  const handleAdd = () => {
    const newText = `新文档 ${db.length + 1}`;
    setDb([...db, { id: db.length + 1, text: newText }]);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">向量数据库操作</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        向量数据库是 RAG 系统的核心组件，支持语义相似度检索。
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-bold text-slate-800 dark:text-slate-100">数据库 ({db.length} 条)</h4>
            <button onClick={handleAdd} className="p-2 bg-rose-100 rounded-lg hover:bg-rose-200">
              <Plus className="w-4 h-4 text-rose-600" />
            </button>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {db.map((item) => (
              <div key={item.id} className="p-2 bg-slate-50 dark:bg-slate-900 rounded text-sm text-slate-700 dark:text-slate-200">
                {item.text}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-4">语义搜索</h4>
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入查询..."
              className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200"
            />
            <button onClick={handleSearch} className="px-4 py-2 bg-rose-600 text-white rounded-lg hover:bg-rose-700">
              <Search className="w-4 h-4" />
            </button>
          </div>
          {results.length > 0 && (
            <div className="space-y-2">
              <span className="text-sm text-slate-500">搜索结果:</span>
              {results.map((item) => (
                <div key={item.id} className="p-2 bg-rose-50 dark:bg-rose-900/20 rounded text-sm">
                  <span className="text-slate-700 dark:text-slate-200">{item.text}</span>
                  <span className="ml-2 text-rose-600 dark:text-rose-400 text-xs">
                    相似度: {(item.similarity! * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

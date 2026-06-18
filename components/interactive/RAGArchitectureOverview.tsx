"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Database, MessageSquare, ArrowRight } from "lucide-react";

const STAGES = [
  { id: "ingest", name: "文档摄取", icon: FileText, items: ["PDF解析", "文本清洗", "分块处理"] },
  { id: "index", name: "向量索引", icon: Database, items: ["Embedding", "向量存储", "元数据"] },
  { id: "retrieve", name: "检索生成", icon: MessageSquare, items: ["语义搜索", "重排序", "答案生成"] },
];

export function RAGArchitectureOverview() {
  const [active, setActive] = useState(0);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">RAG 架构概览</h3>
      <p className="text-slate-600 dark:text-slate-300 mb-6">
        RAG = 检索增强生成。从外部知识库检索信息，增强 LLM 的回答质量。
      </p>

      <div className="flex items-center justify-center gap-4 mb-6">
        {STAGES.map((stage, idx) => {
          const Icon = stage.icon;
          return (
            <React.Fragment key={stage.id}>
              <motion.button
                onClick={() => setActive(idx)}
                animate={{ scale: active === idx ? 1.1 : 1 }}
                className={`w-32 h-32 rounded-2xl flex flex-col items-center justify-center transition-all ${
                  active === idx
                    ? "bg-sky-100 dark:bg-sky-900/30 border-2 border-sky-500"
                    : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
                }`}
              >
                <Icon className={`w-10 h-10 mb-2 ${active === idx ? "text-sky-500" : "text-slate-400"}`} />
                <span className="font-bold text-slate-800 dark:text-slate-100 text-sm">{stage.name}</span>
              </motion.button>
              {idx < STAGES.length - 1 && <ArrowRight className="w-6 h-6 text-slate-300" />}
            </React.Fragment>
          );
        })}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">{STAGES[active].name}</h4>
        <div className="grid grid-cols-3 gap-3">
          {STAGES[active].items.map((item, i) => (
            <div key={i} className="bg-sky-50 dark:bg-sky-900/20 rounded-lg p-3 text-center">
              <span className="text-sm text-sky-700 dark:text-sky-300">{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

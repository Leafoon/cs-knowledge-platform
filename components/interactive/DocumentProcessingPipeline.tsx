"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Scissors, Hash, ArrowRight, CheckCircle } from "lucide-react";

interface ProcessingStage {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  techniques: { name: string; description: string }[];
}

const STAGES: ProcessingStage[] = [
  {
    id: "parse",
    name: "文档解析",
    icon: <FileText className="w-5 h-5" />,
    description: "从各种格式的文档中提取纯文本内容",
    techniques: [
      { name: "PDF解析", description: "使用PyPDF2或pdfplumber提取文本和表格" },
      { name: "HTML解析", description: "使用BeautifulSoup或Trafilatura提取正文" },
      { name: "Word解析", description: "使用python-docx提取段落和样式" },
    ],
  },
  {
    id: "clean",
    name: "文本清洗",
    icon: <Scissors className="w-5 h-5" />,
    description: "去除噪声，标准化文本格式",
    techniques: [
      { name: "去除特殊字符", description: "删除HTML标签、多余空白" },
      { name: "格式标准化", description: "统一编码、换行符" },
      { name: "语言检测", description: "识别文本语言，过滤无关语言" },
    ],
  },
  {
    id: "chunk",
    name: "文本分块",
    icon: <Hash className="w-5 h-5" />,
    description: "将长文本分割成合适大小的块",
    techniques: [
      { name: "固定大小", description: "按字符数或token数分割" },
      { name: "语义分块", description: "基于段落、章节等语义边界" },
      { name: "递归分割", description: "按优先级递归分割长块" },
    ],
  },
];

export function DocumentProcessingPipeline() {
  const [activeStage, setActiveStage] = useState<string>("parse");
  const stage = STAGES.find((s) => s.id === activeStage)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <FileText className="w-6 h-6 text-emerald-500" />
        文档处理流程
      </h3>

      {/* 流程步骤 */}
      <div className="flex items-center justify-center gap-4 mb-8">
        {STAGES.map((s, idx) => (
          <React.Fragment key={s.id}>
            <motion.button
              onClick={() => setActiveStage(s.id)}
              className={`flex items-center gap-2 px-4 py-3 rounded-xl transition-all ${
                activeStage === s.id
                  ? "bg-emerald-100 dark:bg-emerald-900/30 border-2 border-emerald-500"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
              }`}
              whileHover={{ scale: 1.05 }}
            >
              <span className={activeStage === s.id ? "text-emerald-600" : "text-slate-400"}>
                {s.icon}
              </span>
              <span className="font-medium text-slate-700 dark:text-slate-200">{s.name}</span>
            </motion.button>
            {idx < STAGES.length - 1 && (
              <ArrowRight className="w-5 h-5 text-slate-300" />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 阶段详情 */}
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
      >
        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
          {stage.icon}
          {stage.name}
        </h4>
        <p className="text-slate-600 dark:text-slate-300 mb-4">{stage.description}</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stage.techniques.map((tech, i) => (
            <div key={i} className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span className="font-medium text-slate-700 dark:text-slate-200">{tech.name}</span>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-300">{tech.description}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

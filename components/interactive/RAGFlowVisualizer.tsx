"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Database, Search, MessageSquare, ArrowRight, Check } from "lucide-react";

interface FlowStep {
  id: number;
  title: string;
  icon: React.ReactNode;
  description: string;
  details: string[];
}

const FLOW_STEPS: FlowStep[] = [
  {
    id: 1,
    title: "文档加载",
    icon: <FileText className="w-6 h-6" />,
    description: "从各种来源加载原始文档",
    details: ["PDF/Word/HTML解析", "API数据获取", "数据库查询"],
  },
  {
    id: 2,
    title: "文本分块",
    icon: <Database className="w-6 h-6" />,
    description: "将长文档分割为合适大小的块",
    details: ["固定大小分块", "语义分块", "递归字符分割"],
  },
  {
    id: 3,
    title: "向量化",
    icon: <Search className="w-6 h-6" />,
    description: "使用Embedding模型将文本转换为向量",
    details: ["OpenAI Embedding", "本地模型", "批量处理"],
  },
  {
    id: 4,
    title: "检索",
    icon: <Search className="w-6 h-6" />,
    description: "根据用户查询检索相关文档块",
    details: ["相似度搜索", "混合检索", "重排序"],
  },
  {
    id: 5,
    title: "生成",
    icon: <MessageSquare className="w-6 h-6" />,
    description: "基于检索结果生成回答",
    details: ["上下文注入", "提示工程", "答案合成"],
  },
];

export function RAGFlowVisualizer() {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  const handleStepClick = (stepId: number) => {
    setActiveStep(stepId);
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId]);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Database className="w-6 h-6 text-cyan-500" />
        RAG 工作流程
      </h3>

      {/* 流程图 */}
      <div className="flex items-center justify-between mb-8 overflow-x-auto pb-4">
        {FLOW_STEPS.map((step, idx) => (
          <React.Fragment key={step.id}>
            <motion.button
              onClick={() => handleStepClick(step.id)}
              className={`flex flex-col items-center p-4 rounded-xl transition-all min-w-[120px] ${
                activeStep === step.id
                  ? "bg-cyan-100 dark:bg-cyan-900/30 border-2 border-cyan-500 scale-105"
                  : completedSteps.includes(step.id)
                  ? "bg-green-50 dark:bg-green-900/20 border border-green-300"
                  : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-cyan-300"
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className={`mb-2 ${
                activeStep === step.id ? "text-cyan-600" : completedSteps.includes(step.id) ? "text-green-600" : "text-slate-400"
              }`}>
                {completedSteps.includes(step.id) ? <Check className="w-6 h-6" /> : step.icon}
              </span>
              <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{step.title}</span>
            </motion.button>
            {idx < FLOW_STEPS.length - 1 && (
              <ArrowRight className="w-6 h-6 text-slate-300 dark:text-slate-600 flex-shrink-0" />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 步骤详情 */}
      <motion.div
        key={activeStep}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
      >
        <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
          {FLOW_STEPS[activeStep].icon}
          步骤 {FLOW_STEPS[activeStep].id}: {FLOW_STEPS[activeStep].title}
        </h4>
        <p className="text-slate-600 dark:text-slate-300 mb-4">{FLOW_STEPS[activeStep].description}</p>
        <div className="grid grid-cols-3 gap-3">
          {FLOW_STEPS[activeStep].details.map((detail, i) => (
            <div key={i} className="bg-cyan-50 dark:bg-cyan-900/20 rounded-lg p-3 text-center">
              <span className="text-sm text-cyan-700 dark:text-cyan-300">{detail}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

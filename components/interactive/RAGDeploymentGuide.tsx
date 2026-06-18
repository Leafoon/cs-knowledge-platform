"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Server, Cloud, ArrowRight, Check } from "lucide-react";

interface DeploymentOption {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  pros: string[];
  cons: string[];
  bestFor: string;
  costLevel: "low" | "medium" | "high";
}

const OPTIONS: DeploymentOption[] = [
  {
    id: "local",
    name: "本地部署",
    icon: <Server className="w-5 h-5" />,
    description: "在本地机器或服务器上运行RAG系统",
    pros: ["完全控制", "数据安全", "无网络延迟"],
    cons: ["需要硬件", "维护成本", "难以扩展"],
    bestFor: "开发测试、数据敏感场景",
    costLevel: "low",
  },
  {
    id: "cloud",
    name: "云服务部署",
    icon: <Cloud className="w-5 h-5" />,
    description: "使用AWS/Azure/GCP等云服务部署",
    pros: ["弹性扩展", "高可用", "专业运维"],
    cons: ["持续成本", "数据安全", "供应商锁定"],
    bestFor: "生产环境、高并发场景",
    costLevel: "medium",
  },
  {
    id: "saas",
    name: "SaaS平台",
    icon: <Cloud className="w-5 h-5" />,
    description: "使用Pinecone/Weaviate等托管向量数据库",
    pros: ["快速启动", "免运维", "自动扩展"],
    cons: ["成本较高", "定制受限", "数据外传"],
    bestFor: "快速原型、小团队",
    costLevel: "high",
  },
];

export function RAGDeploymentGuide() {
  const [selectedOption, setSelectedOption] = useState<string>("cloud");
  const option = OPTIONS.find((o) => o.id === selectedOption)!;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
        <Server className="w-6 h-6 text-teal-500" />
        RAG 部署方案对比
      </h3>

      {/* 方案选择 */}
      <div className="flex gap-4 mb-6">
        {OPTIONS.map((o) => (
          <motion.button
            key={o.id}
            onClick={() => setSelectedOption(o.id)}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all ${
              selectedOption === o.id
                ? "bg-teal-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-teal-100"
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {o.icon}
            {o.name}
          </motion.button>
        ))}
      </div>

      {/* 方案详情 */}
      <motion.div
        key={option.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700"
      >
        <div className="flex justify-between items-start mb-4">
          <div>
            <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
              {option.icon}
              {option.name}
            </h4>
            <p className="text-slate-600 dark:text-slate-300 mt-2">{option.description}</p>
          </div>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            option.costLevel === "low" ? "bg-green-100 text-green-700" :
            option.costLevel === "medium" ? "bg-yellow-100 text-yellow-700" :
            "bg-red-100 text-red-700"
          }`}>
            成本: {option.costLevel === "low" ? "低" : option.costLevel === "medium" ? "中" : "高"}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">优点</h5>
            <ul className="space-y-2">
              {option.pros.map((p, i) => (
                <li key={i} className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                  <Check className="w-4 h-4 text-green-500" />
                  {p}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h5 className="font-semibold text-red-600 dark:text-red-400 mb-2">缺点</h5>
            <ul className="space-y-2">
              {option.cons.map((c, i) => (
                <li key={i} className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
                  <span className="text-red-500">✗</span>
                  {c}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-4 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg">
          <span className="text-sm font-medium text-teal-700 dark:text-teal-300">最适合: </span>
          <span className="text-teal-600 dark:text-teal-200">{option.bestFor}</span>
        </div>
      </motion.div>
    </div>
  );
}

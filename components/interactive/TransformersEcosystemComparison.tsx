"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { Check, Star, GitBranch, FileText, Users, Clock } from "lucide-react";

interface Framework {
  name: string;
  models: string;
  frameworks: string[];
  easeOfUse: number;
  docs: number;
  adoption: string;
  updateFreq: string;
  color: string;
  recommended?: boolean;
}

const frameworks: Framework[] = [
  {
    name: "Transformers",
    models: "200,000+",
    frameworks: ["PyTorch", "TensorFlow", "JAX"],
    easeOfUse: 5,
    docs: 5,
    adoption: "广泛",
    updateFreq: "每周",
    color: "bg-yellow-500",
    recommended: true
  },
  {
    name: "Fairseq",
    models: "~100",
    frameworks: ["PyTorch"],
    easeOfUse: 3,
    docs: 4,
    adoption: "学术为主",
    updateFreq: "每月",
    color: "bg-blue-500"
  },
  {
    name: "AllenNLP",
    models: "~50",
    frameworks: ["PyTorch"],
    easeOfUse: 4,
    docs: 4,
    adoption: "中等",
    updateFreq: "不定期",
    color: "bg-green-500"
  },
  {
    name: "PaddleNLP",
    models: "500+",
    frameworks: ["PaddlePaddle"],
    easeOfUse: 4,
    docs: 3,
    adoption: "中国市场",
    updateFreq: "每月",
    color: "bg-cyan-500"
  }
];

export default function TransformersEcosystemComparison() {
  const [selectedFramework, setSelectedFramework] = useState<string | null>(null);

  const renderStars = (count: number) => {
    return (
      <div className="flex gap-0.5">
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            className={`w-3.5 h-3.5 ${
              i < count 
                ? "fill-yellow-400 text-yellow-400" 
                : "fill-none text-gray-500"
            }`}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="my-8">
      <div className="mb-6 text-center">
        <h3 className="text-2xl font-semibold text-text-primary mb-2">
          NLP 框架生态对比
        </h3>
        <p className="text-sm text-text-secondary">
          选择最适合你的 NLP 开发框架
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {frameworks.map((fw, index) => (
          <motion.div
            key={fw.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.08 }}
            onClick={() => setSelectedFramework(selectedFramework === fw.name ? null : fw.name)}
            className={`relative bg-white dark:bg-gray-800 rounded-lg border-2 transition-all duration-200 cursor-pointer overflow-hidden ${
              selectedFramework === fw.name
                ? "border-primary shadow-lg scale-[1.02]"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            }`}
          >
            {/* Color accent bar */}
            <div className={`h-1.5 ${fw.color}`} />
            
            {/* Content */}
            <div className="p-5">
              {/* Header */}
              <div className="mb-4">
                <div className="flex items-start justify-between mb-1">
                  <h4 className="text-lg font-semibold text-text-primary">
                    {fw.name}
                  </h4>
                  {fw.recommended && (
                    <span className="flex items-center gap-1 px-2 py-0.5 text-xs font-medium bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded">
                      <Check className="w-3 h-3" />
                      推荐
                    </span>
                  )}
                </div>
              </div>

              {/* Stats */}
              <div className="space-y-3 text-sm">
                {/* Model count */}
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary flex items-center gap-1.5">
                    <GitBranch className="w-3.5 h-3.5" />
                    模型数量
                  </span>
                  <span className="font-semibold text-text-primary">
                    {fw.models}
                  </span>
                </div>

                {/* Frameworks */}
                <div>
                  <div className="text-text-secondary text-xs mb-1.5">支持框架</div>
                  <div className="flex flex-wrap gap-1">
                    {fw.frameworks.map(f => (
                      <span
                        key={f}
                        className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-text-primary rounded"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Ease of use */}
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary text-xs">易用性</span>
                  {renderStars(fw.easeOfUse)}
                </div>

                {/* Documentation */}
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary flex items-center gap-1.5 text-xs">
                    <FileText className="w-3.5 h-3.5" />
                    文档质量
                  </span>
                  {renderStars(fw.docs)}
                </div>

                {/* Adoption */}
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary flex items-center gap-1.5">
                    <Users className="w-3.5 h-3.5" />
                    应用范围
                  </span>
                  <span className="text-text-primary font-medium">
                    {fw.adoption}
                  </span>
                </div>

                {/* Update frequency */}
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    更新频率
                  </span>
                  <span className="text-text-primary font-medium">
                    {fw.updateFreq}
                  </span>
                </div>
              </div>

              {/* Expanded content for Transformers */}
              {selectedFramework === fw.name && fw.recommended && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700"
                >
                  <div className="space-y-2 text-xs text-text-secondary">
                    <div className="flex items-start gap-2">
                      <Check className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>GitHub 120k+ stars</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <Check className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>工业界事实标准</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <Check className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>最活跃的社区支持</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 text-center text-xs text-text-secondary">
        💡 点击卡片查看详细信息
      </div>
    </div>
  );
}

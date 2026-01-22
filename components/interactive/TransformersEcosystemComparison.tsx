"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Framework {
  name: string;
  models: string;
  frameworks: string[];
  easeOfUse: number;
  docs: number;
  adoption: string;
  updateFreq: string;
  color: string;
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
    color: "#FFD21E"
  },
  {
    name: "Fairseq",
    models: "~100",
    frameworks: ["PyTorch"],
    easeOfUse: 3,
    docs: 4,
    adoption: "学术为主",
    updateFreq: "每月",
    color: "#0088CC"
  },
  {
    name: "AllenNLP",
    models: "~50",
    frameworks: ["PyTorch"],
    easeOfUse: 4,
    docs: 4,
    adoption: "中等",
    updateFreq: "不定期",
    color: "#2ECC71"
  },
  {
    name: "PaddleNLP",
    models: "500+",
    frameworks: ["PaddlePaddle"],
    easeOfUse: 4,
    docs: 3,
    adoption: "中国市场",
    updateFreq: "每月",
    color: "#3498DB"
  }
];

export default function TransformersEcosystemComparison() {
  const [selectedFramework, setSelectedFramework] = useState<string | null>(null);

  const renderStars = (count: number) => {
    return (
      <div className="flex gap-0.5">
        {[...Array(5)].map((_, i) => (
          <span
            key={i}
            className={i < count ? "text-yellow-400" : "text-gray-600"}
          >
            ★
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent">
        🤗 框架生态对比
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {frameworks.map((fw, index) => (
          <motion.div
            key={fw.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05, y: -5 }}
            onClick={() => setSelectedFramework(selectedFramework === fw.name ? null : fw.name)}
            className={`p-5 rounded-lg cursor-pointer transition-all duration-300 ${
              selectedFramework === fw.name
                ? "bg-slate-700 shadow-lg ring-2 ring-yellow-400"
                : "bg-slate-800 hover:bg-slate-750"
            }`}
            style={{
              borderLeft: `4px solid ${fw.color}`
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-bold text-white">{fw.name}</h4>
              {fw.name === "Transformers" && (
                <span className="px-2 py-1 text-xs font-bold bg-yellow-500 text-black rounded-full">
                  推荐
                </span>
              )}
            </div>

            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-400">模型数量：</span>
                <span className="text-white font-semibold ml-2">{fw.models}</span>
              </div>

              <div>
                <span className="text-gray-400">支持框架：</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {fw.frameworks.map(f => (
                    <span
                      key={f}
                      className="px-2 py-0.5 bg-slate-600 text-white text-xs rounded"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <span className="text-gray-400 block mb-1">易用性：</span>
                {renderStars(fw.easeOfUse)}
              </div>

              <div>
                <span className="text-gray-400 block mb-1">文档：</span>
                {renderStars(fw.docs)}
              </div>

              <div>
                <span className="text-gray-400">应用：</span>
                <span className="text-white font-semibold ml-2">{fw.adoption}</span>
              </div>

              <div>
                <span className="text-gray-400">更新：</span>
                <span className="text-white font-semibold ml-2">{fw.updateFreq}</span>
              </div>
            </div>

            {selectedFramework === fw.name && fw.name === "Transformers" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-4 pt-4 border-t border-slate-600 text-xs text-gray-300"
              >
                <p className="mb-2">✅ GitHub 120k+ stars</p>
                <p className="mb-2">✅ 工业界事实标准</p>
                <p>✅ 最活跃的社区支持</p>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      <div className="mt-6 text-center text-sm text-gray-400">
        点击卡片查看更多信息
      </div>
    </div>
  );
}

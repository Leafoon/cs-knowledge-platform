"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface EcosystemItem {
  id: string;
  name: string;
  emoji: string;
  description: string;
  type: "core" | "platform";
  color: string;
}

const ecosystemItems: EcosystemItem[] = [
  // Core Libraries
  {
    id: "transformers",
    name: "Transformers",
    emoji: "🤗",
    description: "预训练模型库（200,000+ 模型）",
    type: "core",
    color: "#FFD21E"
  },
  {
    id: "datasets",
    name: "Datasets",
    emoji: "📊",
    description: "数据集加载与预处理（30,000+ 数据集）",
    type: "core",
    color: "#FF6B6B"
  },
  {
    id: "tokenizers",
    name: "Tokenizers",
    emoji: "✂️",
    description: "极速分词器（Rust 实现，10-100x 加速）",
    type: "core",
    color: "#4ECDC4"
  },
  {
    id: "accelerate",
    name: "Accelerate",
    emoji: "⚡",
    description: "分布式训练抽象层（DDP、FSDP、DeepSpeed）",
    type: "core",
    color: "#95E1D3"
  },
  {
    id: "peft",
    name: "PEFT",
    emoji: "🔧",
    description: "参数高效微调（LoRA、QLoRA）",
    type: "core",
    color: "#F38181"
  },
  {
    id: "optimum",
    name: "Optimum",
    emoji: "🚀",
    description: "硬件加速优化（ONNX、Intel、Habana）",
    type: "core",
    color: "#AA96DA"
  },
  {
    id: "diffusers",
    name: "Diffusers",
    emoji: "🎨",
    description: "扩散模型（Stable Diffusion、DALL-E）",
    type: "core",
    color: "#FCBAD3"
  },
  {
    id: "trl",
    name: "TRL",
    emoji: "🎯",
    description: "强化学习（RLHF、DPO）",
    type: "core",
    color: "#FFFFD2"
  },
  // Platform Services
  {
    id: "hub",
    name: "Hub",
    emoji: "☁️",
    description: "模型与数据集托管平台",
    type: "platform",
    color: "#A8E6CF"
  },
  {
    id: "spaces",
    name: "Spaces",
    emoji: "🌐",
    description: "ML 应用托管（Gradio/Streamlit）",
    type: "platform",
    color: "#FFD3B6"
  },
  {
    id: "inference-api",
    name: "Inference API",
    emoji: "⚙️",
    description: "无服务器推理服务",
    type: "platform",
    color: "#FFAAA5"
  },
  {
    id: "autotrain",
    name: "AutoTrain",
    emoji: "🤖",
    description: "无代码训练平台",
    type: "platform",
    color: "#FF8B94"
  }
];

export default function HuggingFaceEcosystemMap() {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  const coreLibs = ecosystemItems.filter(item => item.type === "core");
  const platformServices = ecosystemItems.filter(item => item.type === "platform");

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 rounded-xl border border-purple-500 shadow-2xl">
      <h3 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent">
        🤗 Hugging Face 生态系统全景图
      </h3>

      {/* Core Libraries */}
      <div className="mb-8">
        <h4 className="text-xl font-semibold mb-4 text-yellow-300 flex items-center gap-2">
          <span className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></span>
          核心库 (Core Libraries)
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {coreLibs.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.1, rotate: 2 }}
              onClick={() => setSelectedItem(selectedItem === item.id ? null : item.id)}
              className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                selectedItem === item.id
                  ? "bg-white/20 shadow-xl ring-2 ring-white/50"
                  : "bg-white/10 hover:bg-white/15"
              }`}
              style={{
                backdropFilter: "blur(10px)"
              }}
            >
              <div className="text-4xl mb-2 text-center">{item.emoji}</div>
              <div className="text-sm font-bold text-white text-center mb-1">
                {item.name}
              </div>
              <div className={`text-xs text-gray-300 text-center transition-all ${
                selectedItem === item.id ? "opacity-100" : "opacity-70"
              }`}>
                {item.description}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Platform Services */}
      <div>
        <h4 className="text-xl font-semibold mb-4 text-pink-300 flex items-center gap-2">
          <span className="w-3 h-3 bg-pink-400 rounded-full animate-pulse"></span>
          平台服务 (Platform Services)
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {platformServices.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: (coreLibs.length + index) * 0.1 }}
              whileHover={{ scale: 1.1, rotate: -2 }}
              onClick={() => setSelectedItem(selectedItem === item.id ? null : item.id)}
              className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                selectedItem === item.id
                  ? "bg-white/20 shadow-xl ring-2 ring-white/50"
                  : "bg-white/10 hover:bg-white/15"
              }`}
              style={{
                backdropFilter: "blur(10px)"
              }}
            >
              <div className="text-4xl mb-2 text-center">{item.emoji}</div>
              <div className="text-sm font-bold text-white text-center mb-1">
                {item.name}
              </div>
              <div className={`text-xs text-gray-300 text-center transition-all ${
                selectedItem === item.id ? "opacity-100" : "opacity-70"
              }`}>
                {item.description}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Connection Lines Animation */}
      <div className="mt-8 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="text-sm text-gray-300 bg-white/5 p-3 rounded-lg inline-block"
        >
          💡 点击组件查看详细描述
        </motion.div>
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import { 
  Library, Database, Scissors, Zap, Wrench, Rocket, 
  Palette, Target, Cloud, Globe, Settings, Bot 
} from "lucide-react";

interface EcosystemItem {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  type: "core" | "platform";
}

const ecosystemItems: EcosystemItem[] = [
  // Core Libraries
  {
    id: "transformers",
    name: "Transformers",
    icon: <Library className="w-8 h-8" />,
    description: "预训练模型库（200,000+ 模型）",
    type: "core"
  },
  {
    id: "datasets",
    name: "Datasets",
    icon: <Database className="w-8 h-8" />,
    description: "数据集加载与预处理（30,000+ 数据集）",
    type: "core"
  },
  {
    id: "tokenizers",
    name: "Tokenizers",
    icon: <Scissors className="w-8 h-8" />,
    description: "极速分词器（Rust 实现，10-100x 加速）",
    type: "core"
  },
  {
    id: "accelerate",
    name: "Accelerate",
    icon: <Zap className="w-8 h-8" />,
    description: "分布式训练抽象层（DDP、FSDP、DeepSpeed）",
    type: "core"
  },
  {
    id: "peft",
    name: "PEFT",
    icon: <Wrench className="w-8 h-8" />,
    description: "参数高效微调（LoRA、QLoRA）",
    type: "core"
  },
  {
    id: "optimum",
    name: "Optimum",
    icon: <Rocket className="w-8 h-8" />,
    description: "硬件加速优化（ONNX、Intel、Habana）",
    type: "core"
  },
  {
    id: "diffusers",
    name: "Diffusers",
    icon: <Palette className="w-8 h-8" />,
    description: "扩散模型（Stable Diffusion、DALL-E）",
    type: "core"
  },
  {
    id: "trl",
    name: "TRL",
    icon: <Target className="w-8 h-8" />,
    description: "强化学习（RLHF、DPO）",
    type: "core"
  },
  // Platform Services
  {
    id: "hub",
    name: "Hub",
    icon: <Cloud className="w-8 h-8" />,
    description: "模型与数据集托管平台",
    type: "platform"
  },
  {
    id: "spaces",
    name: "Spaces",
    icon: <Globe className="w-8 h-8" />,
    description: "ML 应用托管（Gradio/Streamlit）",
    type: "platform"
  },
  {
    id: "inference-api",
    name: "Inference API",
    icon: <Settings className="w-8 h-8" />,
    description: "无服务器推理服务",
    type: "platform"
  },
  {
    id: "autotrain",
    name: "AutoTrain",
    icon: <Bot className="w-8 h-8" />,
    description: "无代码训练平台",
    type: "platform"
  }
];

export default function HuggingFaceEcosystemMap() {
  const coreLibs = ecosystemItems.filter(item => item.type === "core");
  const platformServices = ecosystemItems.filter(item => item.type === "platform");

  return (
    <div className="my-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <h3 className="text-2xl font-semibold text-text-primary mb-2">
          Hugging Face 生态系统全景图
        </h3>
        <p className="text-sm text-text-secondary">
          构建现代 AI 应用的完整工具链
        </p>
      </div>

      {/* Core Libraries */}
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-6 bg-blue-500 rounded-full"></div>
          <h4 className="text-lg font-semibold text-text-primary">
            核心库 (Core Libraries)
          </h4>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {coreLibs.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="group relative bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5 hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-md transition-all duration-200"
            >
              {/* Icon */}
              <div className="mb-3 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform duration-200">
                {item.icon}
              </div>
              
              {/* Name */}
              <h5 className="font-semibold text-text-primary mb-2">
                {item.name}
              </h5>
              
              {/* Description */}
              <p className="text-xs text-text-secondary leading-relaxed">
                {item.description}
              </p>

              {/* Hover effect */}
              <div className="absolute inset-0 bg-blue-500/5 dark:bg-blue-400/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"></div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Platform Services */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-1 h-6 bg-purple-500 rounded-full"></div>
          <h4 className="text-lg font-semibold text-text-primary">
            平台服务 (Platform Services)
          </h4>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {platformServices.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: (coreLibs.length + index) * 0.05 }}
              className="group relative bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-5 hover:border-purple-400 dark:hover:border-purple-500 hover:shadow-md transition-all duration-200"
            >
              {/* Icon */}
              <div className="mb-3 text-purple-600 dark:text-purple-400 group-hover:scale-110 transition-transform duration-200">
                {item.icon}
              </div>
              
              {/* Name */}
              <h5 className="font-semibold text-text-primary mb-2">
                {item.name}
              </h5>
              
              {/* Description */}
              <p className="text-xs text-text-secondary leading-relaxed">
                {item.description}
              </p>

              {/* Hover effect */}
              <div className="absolute inset-0 bg-purple-500/5 dark:bg-purple-400/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"></div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

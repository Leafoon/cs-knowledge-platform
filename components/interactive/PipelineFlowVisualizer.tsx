"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Stage {
  id: number;
  name: string;
  emoji: string;
  input: string;
  output: string;
  description: string;
  color: string;
}

const stages: Stage[] = [
  {
    id: 1,
    name: "Tokenization",
    emoji: "✂️",
    input: "I love Transformers!",
    output: "[101, 1045, 2293, 19081, 999, 102]",
    description: "文本转换为 Token IDs，添加特殊 token",
    color: "from-blue-500 to-cyan-500"
  },
  {
    id: 2,
    name: "Model Inference",
    emoji: "🧠",
    input: "[101, 1045, 2293, ...]",
    output: "logits: [-4.23, 4.56]",
    description: "前向传播计算，得到原始分数",
    color: "from-purple-500 to-pink-500"
  },
  {
    id: 3,
    name: "Post-processing",
    emoji: "📊",
    input: "logits: [-4.23, 4.56]",
    output: '{"label": "POSITIVE", "score": 0.9998}',
    description: "应用 softmax，格式化输出",
    color: "from-green-500 to-emerald-500"
  }
];

export default function PipelineFlowVisualizer() {
  const [activeStage, setActiveStage] = useState<number>(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const handleAnimate = () => {
    setIsAnimating(true);
    setActiveStage(0);

    const interval = setInterval(() => {
      setActiveStage((prev) => {
        if (prev >= 2) {
          clearInterval(interval);
          setIsAnimating(false);
          return 0;
        }
        return prev + 1;
      });
    }, 1500);
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
        🔄 Pipeline 三阶段流程
      </h3>

      {/* Visual Flow */}
      <div className="relative mb-8">
        {/* Stages */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {stages.map((stage, index) => (
            <motion.div
              key={stage.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              className="relative"
            >
              {/* Stage Card */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                onClick={() => setActiveStage(index)}
                className={`p-5 rounded-lg cursor-pointer transition-all duration-300 ${
                  activeStage === index
                    ? "bg-gradient-to-br shadow-xl ring-2 ring-white/50"
                    : "bg-slate-800 hover:bg-slate-750"
                } bg-gradient-to-br ${stage.color}`}
                style={{
                  opacity: activeStage === index ? 1 : 0.7
                }}
              >
                <div className="text-4xl mb-2 text-center">{stage.emoji}</div>
                <h4 className="text-lg font-bold text-white text-center mb-2">
                  {stage.name}
                </h4>
                <p className="text-xs text-white/90 text-center mb-3">
                  {stage.description}
                </p>

                {/* Input */}
                <div className="mb-2">
                  <div className="text-xs text-white/70 mb-1">输入:</div>
                  <div className="p-2 bg-black/30 rounded text-xs font-mono text-green-300 break-all">
                    {stage.input}
                  </div>
                </div>

                {/* Arrow */}
                <div className="text-center text-2xl text-white/80 my-1">↓</div>

                {/* Output */}
                <div>
                  <div className="text-xs text-white/70 mb-1">输出:</div>
                  <div className="p-2 bg-black/30 rounded text-xs font-mono text-blue-300 break-all">
                    {stage.output}
                  </div>
                </div>
              </motion.div>

              {/* Connection Arrow (hidden on last item and mobile) */}
              {index < stages.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-8 transform -translate-y-1/2 z-10">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: index * 0.2 + 0.3 }}
                    className="text-4xl text-yellow-400"
                  >
                    →
                  </motion.div>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Animated Data Flow */}
        {isAnimating && (
          <motion.div
            initial={{ x: 0, opacity: 0 }}
            animate={{
              x: ["0%", "33%", "66%", "100%"],
              opacity: [0, 1, 1, 0]
            }}
            transition={{ duration: 4.5, ease: "linear" }}
            className="absolute top-1/2 left-0 transform -translate-y-1/2 hidden md:block"
          >
            <div className="w-8 h-8 bg-yellow-400 rounded-full shadow-lg flex items-center justify-center">
              <span className="text-lg">✨</span>
            </div>
          </motion.div>
        )}
      </div>

      {/* Control Button */}
      <div className="text-center">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleAnimate}
          disabled={isAnimating}
          className={`px-6 py-3 rounded-lg font-semibold text-white transition-all ${
            isAnimating
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg"
          }`}
        >
          {isAnimating ? "动画播放中..." : "▶️ 播放流程动画"}
        </motion.button>
      </div>

      {/* Details Panel */}
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: "auto" }}
        className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700"
      >
        <h4 className="text-sm font-bold text-yellow-400 mb-2">
          当前阶段: {stages[activeStage].name}
        </h4>
        <p className="text-sm text-gray-300">
          {stages[activeStage].description}
        </p>
      </motion.div>
    </div>
  );
}

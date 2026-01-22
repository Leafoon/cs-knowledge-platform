"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

interface Token {
  id: number;
  word: string;
  probability: number;
  rank: number;
}

const generateMockTokens = (count: number): Token[] => {
  const words = [
    "the", "a", "is", "was", "for", "and", "to", "of", "in", "that",
    "it", "with", "as", "on", "be", "at", "by", "this", "from", "or",
    "an", "are", "which", "have", "has", "not", "but", "can", "will", "said"
  ];
  
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    word: words[i % words.length],
    probability: Math.exp(-i * 0.3), // 指数衰减
    rank: i + 1
  }));
};

export default function TopKTopPVisualizer() {
  const [topK, setTopK] = useState(10);
  const [topP, setTopP] = useState(0.9);
  const [mode, setMode] = useState<"topk" | "topp">("topk");
  
  const allTokens = generateMockTokens(30);
  
  // 归一化概率
  const totalProb = allTokens.reduce((sum, t) => sum + t.probability, 0);
  const normalizedTokens = allTokens.map(t => ({
    ...t,
    probability: t.probability / totalProb
  }));

  // Top-K 过滤
  const topKTokens = normalizedTokens.slice(0, topK);

  // Top-P 过滤
  const topPTokens: Token[] = [];
  let cumProb = 0;
  for (const token of normalizedTokens) {
    cumProb += token.probability;
    topPTokens.push(token);
    if (cumProb >= topP) break;
  }

  const selectedTokens = mode === "topk" ? topKTokens : topPTokens;

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-indigo-900 to-purple-900 rounded-xl border border-purple-500 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent">
        🎯 Top-K vs Top-P 采样可视化
      </h3>

      {/* Mode Selector */}
      <div className="flex justify-center gap-4 mb-6">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setMode("topk")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            mode === "topk"
              ? "bg-blue-600 text-white shadow-lg"
              : "bg-white/10 text-gray-300 hover:bg-white/20"
          }`}
        >
          Top-K Sampling
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setMode("topp")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            mode === "topp"
              ? "bg-purple-600 text-white shadow-lg"
              : "bg-white/10 text-gray-300 hover:bg-white/20"
          }`}
        >
          Top-P (Nucleus) Sampling
        </motion.button>
      </div>

      {/* Controls */}
      <div className="mb-8">
        <AnimatePresence mode="wait">
          {mode === "topk" ? (
            <motion.div
              key="topk-control"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-2"
            >
              <div className="flex items-center justify-between">
                <label className="text-sm font-semibold text-white">
                  Top-K 值 (选择前 K 个最高概率的 token)
                </label>
                <span className="px-3 py-1 bg-blue-600 text-white text-sm font-mono rounded">
                  K = {topK}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, 
                    #2563eb 0%, 
                    #2563eb ${(topK / 30) * 100}%, 
                    #334155 ${(topK / 30) * 100}%, 
                    #334155 100%)`
                }}
              />
            </motion.div>
          ) : (
            <motion.div
              key="topp-control"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-2"
            >
              <div className="flex items-center justify-between">
                <label className="text-sm font-semibold text-white">
                  Top-P 值 (累计概率阈值)
                </label>
                <span className="px-3 py-1 bg-purple-600 text-white text-sm font-mono rounded">
                  P = {topP.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, 
                    #9333ea 0%, 
                    #9333ea ${topP * 100}%, 
                    #334155 ${topP * 100}%, 
                    #334155 100%)`
                }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Visualization */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-gray-300">
            概率分布（前 30 个 token）
          </h4>
          <span className="text-xs text-gray-400">
            已选择: {selectedTokens.length} / {normalizedTokens.length}
          </span>
        </div>

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {normalizedTokens.map((token, index) => {
            const isSelected = selectedTokens.some(t => t.id === token.id);
            const barWidth = (token.probability * 100).toFixed(2);

            return (
              <motion.div
                key={token.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.02 }}
                className={`flex items-center gap-3 p-2 rounded transition-all ${
                  isSelected
                    ? mode === "topk"
                      ? "bg-blue-900/40 border border-blue-500"
                      : "bg-purple-900/40 border border-purple-500"
                    : "bg-slate-800/50 border border-transparent"
                }`}
              >
                <span className="text-xs text-gray-400 w-8">#{token.rank}</span>
                <span className="text-sm font-mono text-white w-20">{token.word}</span>
                
                {/* Probability Bar */}
                <div className="flex-1 bg-slate-700 rounded-full h-6 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{ duration: 0.5, delay: index * 0.02 }}
                    className={`h-full flex items-center justify-end pr-2 ${
                      isSelected
                        ? mode === "topk"
                          ? "bg-gradient-to-r from-blue-500 to-blue-600"
                          : "bg-gradient-to-r from-purple-500 to-purple-600"
                        : "bg-gradient-to-r from-gray-500 to-gray-600"
                    }`}
                  >
                    <span className="text-xs font-semibold text-white">
                      {(token.probability * 100).toFixed(1)}%
                    </span>
                  </motion.div>
                </div>

                {isSelected && (
                  <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="text-green-400"
                  >
                    ✓
                  </motion.span>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg border ${
          mode === "topk"
            ? "bg-blue-900/20 border-blue-500/50"
            : "bg-purple-900/20 border-purple-500/50"
        }`}>
          <h5 className="text-sm font-semibold text-white mb-2">📊 统计信息</h5>
          <ul className="text-xs text-gray-300 space-y-1">
            <li>• 候选 token 数: {selectedTokens.length}</li>
            <li>• 累计概率: {(selectedTokens.reduce((sum, t) => sum + t.probability, 0) * 100).toFixed(1)}%</li>
            <li>• 平均概率: {((selectedTokens.reduce((sum, t) => sum + t.probability, 0) / selectedTokens.length) * 100).toFixed(2)}%</li>
          </ul>
        </div>

        <div className="p-4 bg-yellow-900/20 border border-yellow-500/50 rounded-lg">
          <h5 className="text-sm font-semibold text-yellow-400 mb-2">💡 工作原理</h5>
          <p className="text-xs text-gray-300">
            {mode === "topk" 
              ? `Top-K 固定选择前 ${topK} 个最高概率的 token，候选集大小恒定。`
              : `Top-P 动态选择 token，直到累计概率达到 ${(topP * 100).toFixed(0)}%，候选集大小可变。`
            }
          </p>
        </div>
      </div>
    </div>
  );
}

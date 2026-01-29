"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const examples = [
  { input: "happy", output: "sad", embedding: [0.8, 0.2, 0.1] },
  { input: "tall", output: "short", embedding: [0.2, 0.9, 0.1] },
  { input: "hot", output: "cold", embedding: [0.7, 0.1, 0.8] },
  { input: "fast", output: "slow", embedding: [0.1, 0.7, 0.3] },
  { input: "light", output: "dark", embedding: [0.9, 0.3, 0.2] },
  { input: "big", output: "small", embedding: [0.3, 0.8, 0.4] },
];

type SelectorType = 'semantic' | 'mmr' | 'length';

const cosineSimilarity = (a: number[], b: number[]): number => {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magA * magB);
};

export default function FewShotExampleSelector() {
  const [queryInput, setQueryInput] = useState("bright");
  const [selectorType, setSelectorType] = useState<SelectorType>('semantic');
  const [k, setK] = useState(2);
  const [lambdaMult, setLambdaMult] = useState(0.5);

  // 模拟输入的 embedding
  const queryEmbedding = [0.85, 0.25, 0.15];

  // 计算相似度分数
  const scoredExamples = examples.map(ex => ({
    ...ex,
    similarity: cosineSimilarity(ex.embedding, queryEmbedding)
  }));

  // 根据选择器类型筛选
  let selectedExamples: typeof scoredExamples = [];

  if (selectorType === 'semantic') {
    selectedExamples = scoredExamples
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  } else if (selectorType === 'mmr') {
    // 简化的 MMR 实现
    const selected: typeof scoredExamples = [];
    const remaining = [...scoredExamples].sort((a, b) => b.similarity - a.similarity);
    
    while (selected.length < k && remaining.length > 0) {
      const scores = remaining.map((ex, idx) => {
        const relevance = ex.similarity;
        const maxDiversity = selected.length === 0 ? 0 : Math.max(
          ...selected.map(s => cosineSimilarity(ex.embedding, s.embedding))
        );
        return {
          idx,
          score: lambdaMult * relevance - (1 - lambdaMult) * maxDiversity
        };
      });
      
      const best = scores.reduce((max, s) => s.score > max.score ? s : max, scores[0]);
      selected.push(remaining[best.idx]);
      remaining.splice(best.idx, 1);
    }
    
    selectedExamples = selected;
  } else if (selectorType === 'length') {
    // 简化：基于相似度但限制数量
    const maxTokens = 50;
    selectedExamples = [];
    let currentTokens = 0;
    
    for (const ex of scoredExamples.sort((a, b) => b.similarity - a.similarity)) {
      const exampleTokens = (ex.input + ex.output).length / 4; // 简化估算
      if (currentTokens + exampleTokens <= maxTokens) {
        selectedExamples.push(ex);
        currentTokens += exampleTokens;
      }
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        Few-Shot 示例选择器演示
      </h3>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6">
        动态选择最相关的示例，优化提示质量和 token 使用
      </p>

      {/* 控制面板 */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div>
          <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            输入查询
          </label>
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
            placeholder="输入单词..."
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            选择器类型
          </label>
          <select
            value={selectorType}
            onChange={(e) => setSelectorType(e.target.value as SelectorType)}
            className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
          >
            <option value="semantic">SemanticSimilarity（相似度）</option>
            <option value="mmr">MaxMarginalRelevance（多样性）</option>
            <option value="length">LengthBased（长度控制）</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
            选择数量 (k): {k}
          </label>
          <input
            type="range"
            min="1"
            max="4"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {selectorType === 'mmr' && (
          <div>
            <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
              λ (相似度 vs 多样性): {lambdaMult.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={lambdaMult}
              onChange={(e) => setLambdaMult(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-1">
              <span>多样性优先</span>
              <span>相似度优先</span>
            </div>
          </div>
        )}
      </div>

      {/* 可视化区域 */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* 所有示例 */}
        <div>
          <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
            示例库（{examples.length} 个）
          </h4>
          <div className="space-y-2">
            {scoredExamples.map((ex, idx) => {
              const isSelected = selectedExamples.some(s => s.input === ex.input);
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <div className="font-mono text-sm">
                      <span className="text-slate-900 dark:text-white">{ex.input}</span>
                      <span className="text-slate-400 mx-2">→</span>
                      <span className="text-slate-900 dark:text-white">{ex.output}</span>
                    </div>
                    <div className="text-xs font-semibold text-blue-600 dark:text-blue-400">
                      {(ex.similarity * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="mt-1 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${ex.similarity * 100}%` }}
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* 选中的示例 */}
        <div>
          <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
            选中示例（{selectedExamples.length} 个）
          </h4>
          <div className="space-y-3">
            {selectedExamples.map((ex, idx) => (
              <motion.div
                key={idx}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: idx * 0.1 }}
                className="p-4 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center font-bold">
                    {idx + 1}
                  </div>
                  <div className="text-sm font-semibold">
                    相似度: {(ex.similarity * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="font-mono">
                  <div className="text-sm opacity-80">输入：{ex.input}</div>
                  <div className="text-sm opacity-80">输出：{ex.output}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* 生成的提示预览 */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-6 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          生成的 Few-Shot 提示：
        </h4>
        <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto">
          <code>{`Give the antonym of the word:

${selectedExamples.map((ex, idx) => `Input: ${ex.input}\nOutput: ${ex.output}`).join('\n\n')}

Input: ${queryInput}
Output:`}</code>
        </pre>
      </div>

      {/* 说明 */}
      <div className="mt-6 grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
            SemanticSimilarity
          </h5>
          <p className="text-sm text-blue-800 dark:text-blue-200">
            基于向量相似度选择最相关的示例，适合大多数场景
          </p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <h5 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
            MaxMarginalRelevance
          </h5>
          <p className="text-sm text-purple-800 dark:text-purple-200">
            平衡相似度和多样性，避免选择重复内容
          </p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-100 mb-2">
            LengthBased
          </h5>
          <p className="text-sm text-green-800 dark:text-green-200">
            根据 token 限制动态调整示例数量，控制成本
          </p>
        </div>
      </div>
    </div>
  );
}

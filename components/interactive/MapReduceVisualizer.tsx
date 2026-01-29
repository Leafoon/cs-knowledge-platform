"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function MapReduceVisualizer() {
  const [step, setStep] = useState<'split' | 'map' | 'reduce' | 'complete'>('split');
  
  const document = "这是一篇很长的文章...包含大量信息...需要分段处理...";
  const chunks = [
    { id: 1, text: "这是一篇很长的文章...", summary: "文章介绍" },
    { id: 2, text: "包含大量信息...", summary: "信息丰富" },
    { id: 3, text: "需要分段处理...", summary: "分段处理" }
  ];
  const finalSummary = "综合摘要：这是一篇信息丰富的长文章，采用分段处理方式。";

  const handleNext = () => {
    const sequence: typeof step[] = ['split', 'map', 'reduce', 'complete'];
    const currentIndex = sequence.indexOf(step);
    if (currentIndex < sequence.length - 1) {
      setStep(sequence[currentIndex + 1]);
    }
  };

  const handleReset = () => setStep('split');

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        Map-Reduce 执行流程
      </h3>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6">
        演示如何使用 Map-Reduce 模式处理长文档：分割 → 并行处理 → 结果合并
      </p>

      {/* 进度指示器 */}
      <div className="flex items-center justify-between mb-8">
        {['split', 'map', 'reduce', 'complete'].map((s, idx) => (
          <React.Fragment key={s}>
            <div className="flex flex-col items-center">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white ${
                step === s ? 'bg-blue-500 ring-4 ring-blue-200' : 
                ['split', 'map', 'reduce', 'complete'].indexOf(step) > idx ? 'bg-green-500' : 'bg-slate-300'
              }`}>
                {idx + 1}
              </div>
              <span className="mt-2 text-sm font-medium text-slate-700 dark:text-slate-300 capitalize">
                {s === 'split' ? '分割' : s === 'map' ? 'Map' : s === 'reduce' ? 'Reduce' : '完成'}
              </span>
            </div>
            {idx < 3 && (
              <div className={`flex-1 h-1 mx-4 rounded ${
                ['split', 'map', 'reduce', 'complete'].indexOf(step) > idx ? 'bg-green-500' : 'bg-slate-300'
              }`} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 可视化区域 */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-6 mb-6 min-h-[400px]">
        {/* Split 阶段 */}
        {step === 'split' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
              阶段 1: 文档分割
            </h4>
            
            <div className="bg-white dark:bg-slate-900 rounded-lg p-4 border-2 border-blue-500">
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">原始文档</div>
              <div className="font-mono text-sm text-slate-900 dark:text-white">
                {document}
              </div>
            </div>

            <div className="flex items-center justify-center py-4">
              <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>

            <div className="grid grid-cols-3 gap-4">
              {chunks.map((chunk) => (
                <motion.div
                  key={chunk.id}
                  initial={{ y: -20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: chunk.id * 0.2 }}
                  className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-300 dark:border-purple-700"
                >
                  <div className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-2">
                    Chunk {chunk.id}
                  </div>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300">
                    {chunk.text}
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <code className="text-sm text-slate-700 dark:text-slate-300">
                chunks = text_splitter.split_text(document)
              </code>
            </div>
          </motion.div>
        )}

        {/* Map 阶段 */}
        {step === 'map' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
              阶段 2: Map（并行处理）
            </h4>

            <div className="grid grid-cols-3 gap-4 mb-4">
              {chunks.map((chunk) => (
                <div key={chunk.id} className="space-y-2">
                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-300 dark:border-purple-700">
                    <div className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-2">
                      输入 {chunk.id}
                    </div>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300">
                      {chunk.text}
                    </div>
                  </div>

                  <div className="flex justify-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"
                    />
                  </div>

                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5 }}
                    className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border border-green-300 dark:border-green-700"
                  >
                    <div className="text-xs font-semibold text-green-700 dark:text-green-300 mb-2">
                      摘要 {chunk.id}
                    </div>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300">
                      {chunk.summary}
                    </div>
                  </motion.div>
                </div>
              ))}
            </div>

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <code className="text-sm text-slate-700 dark:text-slate-300">
                {`summaries = map_chain.batch([{"text": c} for c in chunks])`}
              </code>
            </div>
          </motion.div>
        )}

        {/* Reduce 阶段 */}
        {step === 'reduce' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
              阶段 3: Reduce（结果合并）
            </h4>

            <div className="flex justify-center gap-4 mb-6">
              {chunks.map((chunk) => (
                <motion.div
                  key={chunk.id}
                  initial={{ y: 0 }}
                  animate={{ y: [0, -20, 0] }}
                  transition={{ delay: chunk.id * 0.2, duration: 0.8 }}
                  className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border border-green-300 dark:border-green-700 w-40"
                >
                  <div className="text-xs font-semibold text-green-700 dark:text-green-300 mb-2">
                    摘要 {chunk.id}
                  </div>
                  <div className="font-mono text-xs text-slate-700 dark:text-slate-300">
                    {chunk.summary}
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="flex justify-center mb-6">
              <div className="text-blue-500 text-4xl">⬇</div>
            </div>

            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-2 border-blue-500 max-w-2xl mx-auto"
            >
              <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-3">
                最终摘要
              </div>
              <div className="text-slate-900 dark:text-white font-medium">
                {finalSummary}
              </div>
            </motion.div>

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <code className="text-sm text-slate-700 dark:text-slate-300">
                {`final = reduce_chain.invoke({"summaries": "\\n".join(summaries)})`}
              </code>
            </div>
          </motion.div>
        )}

        {/* Complete 阶段 */}
        {step === 'complete' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center py-12"
          >
            <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mb-4">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h4 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
              处理完成！
            </h4>
            <p className="text-slate-600 dark:text-slate-400 text-center max-w-md mb-6">
              成功将长文档通过 Map-Reduce 模式处理为简洁摘要。这种方式特别适合处理超过模型上下文限制的文档。
            </p>

            <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-6 max-w-2xl">
              <h5 className="font-semibold text-slate-900 dark:text-white mb-3">性能优势</h5>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span><strong>并行处理</strong>：Map 阶段并发执行，大幅缩短总时间</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span><strong>突破限制</strong>：可处理任意长度的文档</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span><strong>成本优化</strong>：每个 chunk 独立处理，易于缓存和重试</span>
                </li>
              </ul>
            </div>
          </motion.div>
        )}
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-between">
        <button
          onClick={handleReset}
          className="px-6 py-2 rounded-lg font-medium text-sm bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          重置
        </button>
        
        {step !== 'complete' && (
          <button
            onClick={handleNext}
            className="px-6 py-2 rounded-lg font-medium text-sm bg-blue-500 text-white hover:bg-blue-600 transition-colors"
          >
            下一步 →
          </button>
        )}
      </div>
    </div>
  );
}

'use client';

import React, { useState } from 'react';

interface CritiqueItem {
  aspect: string;
  score: number;
  issue: string;
  suggestion: string;
}

interface Iteration {
  version: number;
  output: string;
  overallScore: number;
  critiques: CritiqueItem[];
}

export default function ReflectionLoopVisualizer() {
  const [currentIteration, setCurrentIteration] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const iterations: Iteration[] = [
    {
      version: 1,
      output: '初稿：LangGraph 是一个用于构建有状态的多 Agent 应用的框架...',
      overallScore: 6.0,
      critiques: [
        { aspect: '技术深度', score: 5, issue: '缺少核心概念解释', suggestion: '补充 StateGraph、Checkpoint 等概念' },
        { aspect: '代码示例', score: 4, issue: '没有实际代码', suggestion: '添加完整的代码示例' },
        { aspect: '结构清晰度', score: 8, issue: '结构尚可，但缺少小节', suggestion: '增加二级标题划分' }
      ]
    },
    {
      version: 2,
      output: '改进版：LangGraph 深度解析\n\n## 核心概念\n1. StateGraph: 状态图定义...\n2. Checkpoint: 持久化机制...\n\n## 代码示例\n```python\nfrom langgraph.graph import StateGraph\n...\n```',
      overallScore: 7.5,
      critiques: [
        { aspect: '技术深度', score: 8, issue: '概念讲解到位', suggestion: '可以增加原理分析' },
        { aspect: '代码示例', score: 7, issue: '代码较简单', suggestion: '补充更复杂的实战案例' },
        { aspect: '结构清晰度', score: 8, issue: '结构改善明显', suggestion: '保持' }
      ]
    },
    {
      version: 3,
      output: '最终版：LangGraph 企业级应用指南\n\n## 核心概念与原理\n...\n\n## 完整代码示例\n...\n\n## 生产最佳实践\n...\n\n## 常见问题与调试',
      overallScore: 8.7,
      critiques: [
        { aspect: '技术深度', score: 9, issue: '无', suggestion: '保持' },
        { aspect: '代码示例', score: 9, issue: '无', suggestion: '保持' },
        { aspect: '结构清晰度', score: 8, issue: '无', suggestion: '保持' }
      ]
    }
  ];

  const maxIterations = iterations.length;

  const nextIteration = () => {
    if (currentIteration < maxIterations - 1) {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentIteration(currentIteration + 1);
        setIsAnimating(false);
      }, 800);
    }
  };

  const reset = () => {
    setCurrentIteration(0);
  };

  const current = iterations[currentIteration];
  const isAcceptable = current.overallScore >= 8.0;

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
        Reflection Loop: 自我批评与迭代改进
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        观察 Agent 如何通过自我批评不断改进输出质量
      </p>

      {/* 迭代进度 */}
      <div className="flex items-center justify-center gap-4 mb-8">
        {iterations.map((iter, index) => (
          <React.Fragment key={index}>
            <div className="flex flex-col items-center">
              <div
                className={`w-20 h-20 rounded-full flex flex-col items-center justify-center transition-all ${
                  index === currentIteration
                    ? 'bg-gradient-to-br from-purple-500 to-pink-500 scale-110 shadow-lg'
                    : index < currentIteration
                    ? 'bg-green-500'
                    : 'bg-gray-300 dark:bg-gray-600'
                } ${isAnimating && index === currentIteration ? 'animate-pulse' : ''}`}
              >
                <span className="text-white text-xs font-semibold">版本 {iter.version}</span>
                <span className="text-white text-lg font-bold">{iter.overallScore.toFixed(1)}</span>
              </div>
              {index < currentIteration && iter.overallScore >= 8.0 && (
                <span className="mt-1 text-xs text-green-600 font-semibold">✓ 达标</span>
              )}
            </div>
            {index < iterations.length - 1 && (
              <div className="w-12 h-1 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    index < currentIteration ? 'bg-green-500 w-full' : 'w-0'
                  }`}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 当前输出 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-lg text-gray-800 dark:text-gray-100">
            当前输出（版本 {current.version}）
          </h4>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">总体评分:</span>
            <div className={`px-4 py-2 rounded-full font-bold ${
              isAcceptable 
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
                : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
            }`}>
              {current.overallScore.toFixed(1)} / 10
            </div>
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
          {current.output}
        </div>
      </div>

      {/* 批评意见 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6 shadow-lg">
        <h4 className="font-bold text-lg mb-4 text-gray-800 dark:text-gray-100">
          批评意见与改进建议
        </h4>
        <div className="space-y-4">
          {current.critiques.map((critique, index) => (
            <div
              key={index}
              className="border-l-4 border-purple-500 pl-4 py-2"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-gray-800 dark:text-gray-100">
                  {critique.aspect}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        critique.score >= 8 ? 'bg-green-500' : critique.score >= 6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${critique.score * 10}%` }}
                    />
                  </div>
                  <span className="text-sm font-bold text-gray-700 dark:text-gray-300">
                    {critique.score}/10
                  </span>
                </div>
              </div>
              {critique.issue !== '无' && (
                <div className="text-sm text-red-600 dark:text-red-400 mb-1">
                  ⚠️ 问题: {critique.issue}
                </div>
              )}
              <div className="text-sm text-blue-600 dark:text-blue-400">
                💡 建议: {critique.suggestion}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex items-center justify-between">
        <div className="flex gap-3">
          <button
            onClick={nextIteration}
            disabled={currentIteration >= maxIterations - 1 || isAnimating}
            className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {currentIteration === 0 ? '🔄 开始改进' : '➡️ 下一次迭代'}
          </button>
          
          <button
            onClick={reset}
            className="px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
          >
            🔁 重置
          </button>
        </div>

        {isAcceptable && (
          <div className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
            <span className="text-2xl">🎉</span>
            <span className="font-semibold text-green-700 dark:text-green-400">
              质量达标！
            </span>
          </div>
        )}
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl border-l-4 border-purple-500">
        <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">💡 Reflection 机制</h4>
        <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
          <li><strong>生成:</strong> Agent 生成初始输出</li>
          <li><strong>批评:</strong> Critic Agent 从多个维度评估质量</li>
          <li><strong>改进:</strong> 根据批评意见重新生成</li>
          <li><strong>迭代:</strong> 重复上述过程直到达到质量标准（≥8.0分）</li>
        </ul>
      </div>
    </div>
  );
}

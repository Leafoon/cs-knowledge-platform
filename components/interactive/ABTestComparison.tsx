"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Target, DollarSign, Zap, Award } from 'lucide-react';

interface Variant {
  name: string;
  description: string;
  avgScore: number;
  scores: number[];
  latency: number;
  cost: number;
  successRate: number;
}

const ABTestComparison: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<'score' | 'latency' | 'cost'>('score');

  const variantA: Variant = {
    name: 'Prompt V1 (简单)',
    description: 'Translate to French: {text}',
    avgScore: 0.75,
    scores: [0.8, 0.75, 0.7, 0.72, 0.78, 0.76, 0.74, 0.75, 0.73, 0.77],
    latency: 850,
    cost: 0.015,
    successRate: 94,
  };

  const variantB: Variant = {
    name: 'Prompt V2 (详细)',
    description: 'Professional translator with cultural context...',
    avgScore: 0.89,
    scores: [0.92, 0.88, 0.86, 0.90, 0.91, 0.87, 0.89, 0.88, 0.90, 0.89],
    latency: 1200,
    cost: 0.028,
    successRate: 98,
  };

  // 计算统计显著性
  const calculatePValue = (scoresA: number[], scoresB: number[]): number => {
    const meanA = scoresA.reduce((a, b) => a + b) / scoresA.length;
    const meanB = scoresB.reduce((a, b) => a + b) / scoresB.length;
    const stdA = Math.sqrt(scoresA.reduce((sum, x) => sum + Math.pow(x - meanA, 2), 0) / scoresA.length);
    const stdB = Math.sqrt(scoresB.reduce((sum, x) => sum + Math.pow(x - meanB, 2), 0) / scoresB.length);
    
    // 简化的 t-test 近似
    const t = Math.abs(meanA - meanB) / Math.sqrt((stdA * stdA + stdB * stdB) / 2);
    
    // 模拟 p-value（实际应用应使用统计库）
    return t > 2 ? 0.012 : 0.156;
  };

  const pValue = calculatePValue(variantA.scores, variantB.scores);
  const isSignificant = pValue < 0.05;

  const improvement = {
    score: ((variantB.avgScore - variantA.avgScore) / variantA.avgScore * 100).toFixed(1),
    latency: ((variantB.latency - variantA.latency) / variantA.latency * 100).toFixed(1),
    cost: ((variantB.cost - variantA.cost) / variantA.cost * 100).toFixed(1),
  };

  const getWinner = () => {
    if (!isSignificant) return 'none';
    return variantB.avgScore > variantA.avgScore ? 'B' : 'A';
  };

  const winner = getWinner();

  const MetricCard: React.FC<{ variant: Variant; isWinner: boolean }> = ({ variant, isWinner }) => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`p-6 rounded-lg border-2 ${
        isWinner ? 'border-green-500 bg-green-50' : 'border-gray-300 bg-white'
      } relative`}
    >
      {isWinner && (
        <div className="absolute -top-3 -right-3 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-bold flex items-center gap-1">
          <Award className="w-4 h-4" />
          获胜
        </div>
      )}
      
      <h3 className="text-lg font-bold text-gray-800 mb-1">{variant.name}</h3>
      <p className="text-sm text-gray-500 mb-4 truncate">{variant.description}</p>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 bg-white rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-4 h-4 text-indigo-500" />
            <p className="text-xs text-gray-600">平均分数</p>
          </div>
          <p className="text-2xl font-bold text-gray-800">{variant.avgScore.toFixed(2)}</p>
        </div>
        <div className="p-3 bg-white rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-4 h-4 text-yellow-500" />
            <p className="text-xs text-gray-600">延迟</p>
          </div>
          <p className="text-2xl font-bold text-gray-800">{variant.latency}<span className="text-sm text-gray-500">ms</span></p>
        </div>
        <div className="p-3 bg-white rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-500" />
            <p className="text-xs text-gray-600">成本</p>
          </div>
          <p className="text-2xl font-bold text-gray-800">${variant.cost.toFixed(3)}</p>
        </div>
        <div className="p-3 bg-white rounded border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-blue-500" />
            <p className="text-xs text-gray-600">成功率</p>
          </div>
          <p className="text-2xl font-bold text-gray-800">{variant.successRate}<span className="text-sm text-gray-500">%</span></p>
        </div>
      </div>

      {/* 分数分布 */}
      <div className="mt-4">
        <p className="text-xs text-gray-600 mb-2">分数分布</p>
        <div className="flex gap-1">
          {variant.scores.map((score, idx) => (
            <div
              key={idx}
              className="flex-1 bg-gray-200 rounded"
              style={{ height: `${score * 60}px` }}
              title={`样本 ${idx + 1}: ${score.toFixed(2)}`}
            >
              <div
                className={`w-full ${isWinner ? 'bg-green-500' : 'bg-indigo-500'} rounded transition-all`}
                style={{ height: '100%' }}
              />
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2">A/B 测试对比分析</h3>
        <p className="text-gray-600">对比两个提示版本的性能，基于数据做出决策</p>
      </div>

      {/* 变体对比 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <MetricCard variant={variantA} isWinner={winner === 'A'} />
        <MetricCard variant={variantB} isWinner={winner === 'B'} />
      </div>

      {/* 改进指标 */}
      <div className="mb-6 p-6 bg-white rounded-lg shadow border border-gray-200">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">📊 改进分析</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">分数提升</p>
            <div className="flex items-center justify-center gap-1">
              {parseFloat(improvement.score) > 0 ? (
                <TrendingUp className="w-5 h-5 text-green-500" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-500" />
              )}
              <p className={`text-2xl font-bold ${parseFloat(improvement.score) > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {improvement.score > '0' ? '+' : ''}{improvement.score}%
              </p>
            </div>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">延迟变化</p>
            <div className="flex items-center justify-center gap-1">
              {parseFloat(improvement.latency) > 0 ? (
                <TrendingUp className="w-5 h-5 text-red-500" />
              ) : (
                <TrendingDown className="w-5 h-5 text-green-500" />
              )}
              <p className={`text-2xl font-bold ${parseFloat(improvement.latency) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {improvement.latency > '0' ? '+' : ''}{improvement.latency}%
              </p>
            </div>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">成本变化</p>
            <div className="flex items-center justify-center gap-1">
              {parseFloat(improvement.cost) > 0 ? (
                <TrendingUp className="w-5 h-5 text-red-500" />
              ) : (
                <TrendingDown className="w-5 h-5 text-green-500" />
              )}
              <p className={`text-2xl font-bold ${parseFloat(improvement.cost) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {improvement.cost > '0' ? '+' : ''}{improvement.cost}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 统计显著性 */}
      <div className={`p-6 rounded-lg border-2 ${isSignificant ? 'bg-green-50 border-green-500' : 'bg-yellow-50 border-yellow-500'}`}>
        <h4 className="text-lg font-semibold text-gray-800 mb-2">
          {isSignificant ? '✅ 结果具有统计显著性' : '⚠️ 结果不具有统计显著性'}
        </h4>
        <p className="text-sm text-gray-700 mb-3">
          p-value = {pValue.toFixed(3)} {isSignificant ? '< 0.05' : '≥ 0.05'}
        </p>
        <div className="bg-white p-4 rounded border border-gray-200">
          {isSignificant ? (
            <div>
              <p className="text-sm text-gray-700 mb-2">
                <strong>✅ 建议：</strong>Variant B 的改进<strong>具有统计学意义</strong>，可以考虑部署。
              </p>
              <ul className="text-sm text-gray-600 space-y-1 ml-4">
                <li>• 分数提升 {improvement.score}%（从 {variantA.avgScore.toFixed(2)} 到 {variantB.avgScore.toFixed(2)}）</li>
                <li>• 权衡：延迟增加 {improvement.latency}%，成本增加 {improvement.cost}%</li>
                <li>• 建议：若用户对质量要求高，部署 V2；若对成本敏感，继续优化</li>
              </ul>
            </div>
          ) : (
            <div>
              <p className="text-sm text-gray-700 mb-2">
                <strong>⚠️ 建议：</strong>差异可能是随机波动，需要更多数据。
              </p>
              <ul className="text-sm text-gray-600 space-y-1 ml-4">
                <li>• 增加测试样本数（当前 10 条，建议 50-100 条）</li>
                <li>• 检查两个变体的配置是否真的不同</li>
                <li>• 考虑使用更严格的评估器</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-purple-200">
        <h4 className="font-semibold text-gray-800 mb-2">💡 A/B 测试最佳实践</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>样本量</strong>：至少 50 条测试样本，确保代表性</li>
          <li>• <strong>统计显著性</strong>：p-value &lt; 0.05 才能说明改进不是偶然</li>
          <li>• <strong>权衡分析</strong>：不仅看分数，还要考虑延迟、成本、可靠性</li>
          <li>• <strong>渐进式部署</strong>：即使 V2 获胜，也应先灰度测试（10% → 50% → 100%）</li>
        </ul>
      </div>
    </div>
  );
};

export default ABTestComparison;

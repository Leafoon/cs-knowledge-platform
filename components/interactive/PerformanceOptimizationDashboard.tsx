'use client';

import React, { useState } from 'react';

type OptimizationTechnique = 'baseline' | 'cache' | 'async' | 'batch' | 'all';

interface PerformanceMetrics {
  technique: string;
  latency: number; // ms
  throughput: number; // requests/sec
  costPerRequest: number; // USD
  cacheHitRate: number; // %
  description: string;
  improvements: {
    latency: number; // % improvement
    cost: number;
    throughput: number;
  };
}

const PerformanceOptimizationDashboard: React.FC = () => {
  const [selectedTechnique, setSelectedTechnique] = useState<OptimizationTechnique>('baseline');

  const metrics: Record<OptimizationTechnique, PerformanceMetrics> = {
    baseline: {
      technique: '基线（无优化）',
      latency: 2500,
      throughput: 10,
      costPerRequest: 0.03,
      cacheHitRate: 0,
      description: '同步调用，无缓存，单请求处理',
      improvements: { latency: 0, cost: 0, throughput: 0 },
    },
    cache: {
      technique: 'Redis 缓存',
      latency: 800,
      throughput: 15,
      costPerRequest: 0.012,
      cacheHitRate: 65,
      description: '启用 L1 内存 + L2 Redis 缓存，7天过期',
      improvements: { latency: 68, cost: 60, throughput: 50 },
    },
    async: {
      technique: '异步并发',
      latency: 2200,
      throughput: 45,
      costPerRequest: 0.028,
      cacheHitRate: 0,
      description: '异步并发执行，连接池优化',
      improvements: { latency: 12, cost: 7, throughput: 350 },
    },
    batch: {
      technique: '批处理',
      latency: 1800,
      throughput: 30,
      costPerRequest: 0.015,
      cacheHitRate: 0,
      description: 'Embedding 批量调用，减少网络开销',
      improvements: { latency: 28, cost: 50, throughput: 200 },
    },
    all: {
      technique: '全套优化',
      latency: 400,
      throughput: 80,
      costPerRequest: 0.006,
      cacheHitRate: 70,
      description: '缓存 + 异步 + 批处理 + 流式',
      improvements: { latency: 84, cost: 80, throughput: 700 },
    },
  };

  const currentMetrics = metrics[selectedTechnique];

  const getProgressBarColor = (value: number, max: number): string => {
    const percentage = (value / max) * 100;
    if (percentage > 80) return 'bg-red-500';
    if (percentage > 50) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const optimizationTechniques = [
    { id: 'cache' as OptimizationTechnique, name: '🗄️ 缓存层级', color: 'from-blue-500 to-cyan-500' },
    { id: 'async' as OptimizationTechnique, name: '⚡ 异步并发', color: 'from-purple-500 to-pink-500' },
    { id: 'batch' as OptimizationTechnique, name: '📦 批处理', color: 'from-green-500 to-teal-500' },
    { id: 'all' as OptimizationTechnique, name: '🚀 全套优化', color: 'from-orange-500 to-red-500' },
  ];

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-gray-800 mb-6">性能优化仪表板</h3>

      {/* 技术选择器 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        <button
          onClick={() => setSelectedTechnique('baseline')}
          className={`p-4 rounded-lg font-semibold transition-all ${
            selectedTechnique === 'baseline'
              ? 'bg-gradient-to-r from-gray-600 to-gray-700 text-white shadow-lg scale-105'
              : 'bg-white text-gray-700 hover:shadow-md'
          }`}
        >
          📊 基线
        </button>
        
        {optimizationTechniques.map((tech) => (
          <button
            key={tech.id}
            onClick={() => setSelectedTechnique(tech.id)}
            className={`p-4 rounded-lg font-semibold transition-all ${
              selectedTechnique === tech.id
                ? `bg-gradient-to-r ${tech.color} text-white shadow-lg scale-105`
                : 'bg-white text-gray-700 hover:shadow-md'
            }`}
          >
            {tech.name}
          </button>
        ))}
      </div>

      {/* 当前技术说明 */}
      <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-xl font-bold text-gray-800">{currentMetrics.technique}</h4>
          {selectedTechnique !== 'baseline' && (
            <div className="flex space-x-3">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                ↓ 延迟 {currentMetrics.improvements.latency}%
              </span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                ↓ 成本 {currentMetrics.improvements.cost}%
              </span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                ↑ 吞吐 {currentMetrics.improvements.throughput}%
              </span>
            </div>
          )}
        </div>
        <p className="text-gray-600">{currentMetrics.description}</p>
      </div>

      {/* 核心指标 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* 延迟 */}
        <div className="bg-white rounded-lg p-5 shadow">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600">平均延迟</span>
            <span className="text-2xl">⏱️</span>
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {currentMetrics.latency}ms
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${getProgressBarColor(currentMetrics.latency, 3000)}`}
              style={{ width: `${(currentMetrics.latency / 3000) * 100}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-2">目标: &lt;500ms</p>
        </div>

        {/* 吞吐量 */}
        <div className="bg-white rounded-lg p-5 shadow">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600">吞吐量</span>
            <span className="text-2xl">🚀</span>
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {currentMetrics.throughput} req/s
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-500"
              style={{ width: `${(currentMetrics.throughput / 100) * 100}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-2">目标: &gt;50 req/s</p>
        </div>

        {/* 成本 */}
        <div className="bg-white rounded-lg p-5 shadow">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600">单请求成本</span>
            <span className="text-2xl">💰</span>
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            ${currentMetrics.costPerRequest.toFixed(3)}
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${getProgressBarColor(currentMetrics.costPerRequest * 1000, 30)}`}
              style={{ width: `${(currentMetrics.costPerRequest / 0.04) * 100}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-2">目标: &lt;$0.01</p>
        </div>

        {/* 缓存命中率 */}
        <div className="bg-white rounded-lg p-5 shadow">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600">缓存命中率</span>
            <span className="text-2xl">🎯</span>
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {currentMetrics.cacheHitRate}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-500"
              style={{ width: `${currentMetrics.cacheHitRate}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-2">目标: &gt;60%</p>
        </div>
      </div>

      {/* 性能对比图 */}
      <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">优化效果对比</h4>
        
        <div className="space-y-4">
          {Object.entries(metrics).map(([key, metric]) => (
            <div
              key={key}
              className={`p-4 rounded-lg transition-all ${
                selectedTechnique === key ? 'bg-blue-50 border-2 border-blue-300' : 'bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-gray-800">{metric.technique}</span>
                <div className="flex space-x-4 text-sm">
                  <span className="text-gray-600">{metric.latency}ms</span>
                  <span className="text-gray-600">${metric.costPerRequest.toFixed(3)}</span>
                  <span className="text-gray-600">{metric.throughput} req/s</span>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-yellow-500 transition-all duration-300"
                      style={{ width: `${100 - (metric.latency / 2500) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">延迟优化</p>
                </div>
                
                <div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-green-500 transition-all duration-300"
                      style={{ width: `${100 - (metric.costPerRequest / 0.03) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">成本优化</p>
                </div>
                
                <div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-blue-500 transition-all duration-300"
                      style={{ width: `${(metric.throughput / 80) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">吞吐提升</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 优化建议 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg p-5 shadow">
          <h5 className="font-semibold text-gray-800 mb-3 flex items-center">
            <span className="mr-2">💡</span>
            快速优化建议
          </h5>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>✓ 启用 Redis 缓存可立即降低 60% 成本</li>
            <li>✓ 异步并发可提升 350% 吞吐量</li>
            <li>✓ Embedding 批处理减少 50% API 调用</li>
            <li>✓ 组合优化可实现 84% 延迟降低</li>
          </ul>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-5 shadow">
          <h5 className="font-semibold text-gray-800 mb-3 flex items-center">
            <span className="mr-2">⚠️</span>
            注意事项
          </h5>
          <ul className="space-y-2 text-sm text-gray-700">
            <li>• 缓存适用于重复查询场景</li>
            <li>• 异步需注意 API 速率限制</li>
            <li>• 批处理增加单次请求延迟</li>
            <li>• 监控缓存命中率和过期策略</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PerformanceOptimizationDashboard;

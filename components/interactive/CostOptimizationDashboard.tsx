"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { DollarSign, Zap, TrendingDown, Settings, BarChart3, AlertTriangle } from 'lucide-react';

type ModelConfig = {
  name: string;
  inputCost: number;  // per 1M tokens
  outputCost: number;
  latency: number;  // ms
  quality: number;  // 0-100
};

const models: Record<string, ModelConfig> = {
  'gpt-4o': {
    name: 'GPT-4o',
    inputCost: 2.50,
    outputCost: 10.00,
    latency: 1500,
    quality: 95
  },
  'gpt-4o-mini': {
    name: 'GPT-4o Mini',
    inputCost: 0.15,
    outputCost: 0.60,
    latency: 800,
    quality: 85
  },
  'gpt-3.5-turbo': {
    name: 'GPT-3.5 Turbo',
    inputCost: 0.50,
    outputCost: 1.50,
    latency: 600,
    quality: 75
  }
};

export default function CostOptimizationDashboard() {
  const [trafficDistribution, setTrafficDistribution] = useState({
    'gpt-4o': 30,
    'gpt-4o-mini': 50,
    'gpt-3.5-turbo': 20
  });

  const [cacheHitRate, setCacheHitRate] = useState(60);
  const [dailyRequests, setDailyRequests] = useState(10000);
  const [avgInputTokens, setAvgInputTokens] = useState(500);
  const [avgOutputTokens, setAvgOutputTokens] = useState(200);

  const calculateCosts = () => {
    let totalCost = 0;
    let totalLatency = 0;
    let avgQuality = 0;

    Object.entries(trafficDistribution).forEach(([model, percentage]) => {
      const config = models[model];
      const requests = dailyRequests * (percentage / 100) * ((100 - cacheHitRate) / 100);
      
      const inputCost = (requests * avgInputTokens / 1000000) * config.inputCost;
      const outputCost = (requests * avgOutputTokens / 1000000) * config.outputCost;
      
      totalCost += inputCost + outputCost;
      totalLatency += config.latency * (percentage / 100);
      avgQuality += config.quality * (percentage / 100);
    });

    return {
      dailyCost: totalCost,
      monthlyCost: totalCost * 30,
      avgLatency: totalLatency,
      avgQuality,
      cachedRequests: dailyRequests * (cacheHitRate / 100),
      llmRequests: dailyRequests * ((100 - cacheHitRate) / 100)
    };
  };

  const metrics = calculateCosts();

  const optimizationStrategies = [
    {
      title: '提高缓存命中率',
      impact: '+15%',
      description: '从 60% 提升到 75%',
      savings: ((metrics.dailyCost * 0.15)).toFixed(2),
      difficulty: '中等',
      implementation: ['启用语义缓存', '预热常见问题', '延长 TTL']
    },
    {
      title: '优化模型路由',
      impact: '+25%',
      description: '简单任务用 gpt-4o-mini',
      savings: (metrics.dailyCost * 0.25).toFixed(2),
      difficulty: '低',
      implementation: ['按长度路由', '按复杂度路由', 'A/B 测试验证']
    },
    {
      title: '批处理请求',
      impact: '+10%',
      description: '合并批量调用',
      savings: (metrics.dailyCost * 0.10).toFixed(2),
      difficulty: '低',
      implementation: ['使用 .batch()', '调整批大小', '异步处理']
    },
    {
      title: '输出长度限制',
      impact: '+8%',
      description: 'max_tokens 限制',
      savings: (metrics.dailyCost * 0.08).toFixed(2),
      difficulty: '低',
      implementation: ['设置 max_tokens', '优化 prompt', '后处理截断']
    }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">成本优化仪表盘</h3>
        <p className="text-gray-600">实时模拟不同优化策略的成本节省效果</p>
      </div>

      {/* 核心指标卡片 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <motion.div
          className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-6 rounded-lg"
          whileHover={{ scale: 1.05 }}
        >
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="w-8 h-8" />
            <span className="text-sm opacity-90">每日成本</span>
          </div>
          <div className="text-3xl font-bold">${metrics.dailyCost.toFixed(2)}</div>
          <div className="text-sm opacity-75 mt-1">月成本: ${metrics.monthlyCost.toFixed(2)}</div>
        </motion.div>

        <motion.div
          className="bg-gradient-to-br from-green-500 to-green-600 text-white p-6 rounded-lg"
          whileHover={{ scale: 1.05 }}
        >
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-8 h-8" />
            <span className="text-sm opacity-90">平均延迟</span>
          </div>
          <div className="text-3xl font-bold">{metrics.avgLatency.toFixed(0)}ms</div>
          <div className="text-sm opacity-75 mt-1">缓存命中: {cacheHitRate}%</div>
        </motion.div>

        <motion.div
          className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-6 rounded-lg"
          whileHover={{ scale: 1.05 }}
        >
          <div className="flex items-center justify-between mb-2">
            <BarChart3 className="w-8 h-8" />
            <span className="text-sm opacity-90">请求总数</span>
          </div>
          <div className="text-3xl font-bold">{dailyRequests.toLocaleString()}</div>
          <div className="text-sm opacity-75 mt-1">
            LLM: {metrics.llmRequests.toLocaleString()}
          </div>
        </motion.div>

        <motion.div
          className="bg-gradient-to-br from-orange-500 to-orange-600 text-white p-6 rounded-lg"
          whileHover={{ scale: 1.05 }}
        >
          <div className="flex items-center justify-between mb-2">
            <TrendingDown className="w-8 h-8" />
            <span className="text-sm opacity-90">平均质量</span>
          </div>
          <div className="text-3xl font-bold">{metrics.avgQuality.toFixed(0)}/100</div>
          <div className="text-sm opacity-75 mt-1">质量得分</div>
        </motion.div>
      </div>

      {/* 配置面板 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 左侧：流量分配 */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            模型流量分配
          </h4>
          
          {Object.entries(trafficDistribution).map(([model, percentage]) => (
            <div key={model} className="mb-4">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">{models[model].name}</span>
                <span className="text-sm font-bold text-blue-600">{percentage}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={percentage}
                onChange={(e) => {
                  const newValue = Number(e.target.value);
                  const otherModels = Object.keys(trafficDistribution).filter(m => m !== model) as Array<keyof typeof trafficDistribution>;
                  const remaining = 100 - newValue;
                  const evenSplit = remaining / otherModels.length;
                  
                  const newDistribution: typeof trafficDistribution = {
                    'gpt-4o': 0,
                    'gpt-4o-mini': 0,
                    'gpt-3.5-turbo': 0
                  };
                  
                  otherModels.forEach(m => {
                    newDistribution[m] = evenSplit;
                  });
                  newDistribution[model as keyof typeof trafficDistribution] = newValue;
                  
                  setTrafficDistribution(newDistribution);
                }}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>成本: ${models[model].inputCost}/1M</span>
                <span>质量: {models[model].quality}/100</span>
              </div>
            </div>
          ))}
        </div>

        {/* 右侧：系统参数 */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5" />
            系统参数
          </h4>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                缓存命中率: {cacheHitRate}%
              </label>
              <input
                type="range"
                min="0"
                max="95"
                value={cacheHitRate}
                onChange={(e) => setCacheHitRate(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                每日请求数: {dailyRequests.toLocaleString()}
              </label>
              <input
                type="range"
                min="1000"
                max="100000"
                step="1000"
                value={dailyRequests}
                onChange={(e) => setDailyRequests(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                平均输入 Tokens: {avgInputTokens}
              </label>
              <input
                type="range"
                min="100"
                max="2000"
                step="50"
                value={avgInputTokens}
                onChange={(e) => setAvgInputTokens(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                平均输出 Tokens: {avgOutputTokens}
              </label>
              <input
                type="range"
                min="50"
                max="1000"
                step="25"
                value={avgOutputTokens}
                onChange={(e) => setAvgOutputTokens(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* 优化建议 */}
      <div className="space-y-3">
        <h4 className="font-semibold flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-600" />
          优化策略建议
        </h4>

        {optimizationStrategies.map((strategy, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="border-2 border-gray-200 rounded-lg p-4 hover:border-blue-400 transition-colors"
          >
            <div className="flex justify-between items-start mb-2">
              <div>
                <h5 className="font-semibold">{strategy.title}</h5>
                <p className="text-sm text-gray-600">{strategy.description}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-green-600">${strategy.savings}/天</div>
                <div className="text-xs text-gray-600">节省 {strategy.impact}</div>
              </div>
            </div>

            <div className="flex items-center gap-2 mb-2">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                strategy.difficulty === '低' ? 'bg-green-100 text-green-700' :
                strategy.difficulty === '中等' ? 'bg-yellow-100 text-yellow-700' :
                'bg-red-100 text-red-700'
              }`}>
                难度: {strategy.difficulty}
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              {strategy.implementation.map((step, i) => (
                <span key={i} className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded">
                  {step}
                </span>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* 总节省预估 */}
      <motion.div
        className="mt-6 bg-gradient-to-r from-green-500 to-green-600 text-white p-6 rounded-lg"
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
      >
        <div className="flex justify-between items-center">
          <div>
            <h4 className="text-lg font-semibold mb-1">实施所有优化后预计节省</h4>
            <p className="text-sm opacity-90">基于当前配置的理论最大值</p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold">
              ${(optimizationStrategies.reduce((sum, s) => sum + Number(s.savings), 0)).toFixed(2)}/天
            </div>
            <div className="text-lg opacity-90">
              ${(optimizationStrategies.reduce((sum, s) => sum + Number(s.savings), 0) * 30).toFixed(2)}/月
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

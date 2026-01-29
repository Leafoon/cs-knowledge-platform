"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Database, Clock, TrendingUp, DollarSign } from 'lucide-react';

type CachingStrategy = {
  id: string;
  name: string;
  icon: React.ReactNode;
  hitRate: number;
  latency: number;
  cost: number;
  ttl: number;
  persistence: boolean;
  distributed: boolean;
  description: string;
  pros: string[];
  cons: string[];
  bestFor: string[];
};

const strategies: CachingStrategy[] = [
  {
    id: 'none',
    name: '无缓存',
    icon: <Zap className="w-6 h-6" />,
    hitRate: 0,
    latency: 1200,
    cost: 100,
    ttl: 0,
    persistence: false,
    distributed: false,
    description: '每次请求都调用 LLM API，无任何缓存机制',
    pros: ['实现简单', '结果始终最新', '无缓存一致性问题'],
    cons: ['成本高昂', '延迟大', '无容错能力', 'API 限流风险'],
    bestFor: ['开发初期快速验证', '结果需要实时性的场景']
  },
  {
    id: 'memory',
    name: 'InMemoryCache',
    icon: <Zap className="w-6 h-6" />,
    hitRate: 45,
    latency: 2,
    cost: 55,
    ttl: 3600,
    persistence: false,
    distributed: false,
    description: '将结果存储在进程内存中，速度极快但无持久化',
    pros: ['延迟极低 (<5ms)', '无外部依赖', '实现简单'],
    cons: ['进程重启丢失', '无法跨实例共享', '内存占用'],
    bestFor: ['本地开发测试', '单机短期服务', '演示 Demo']
  },
  {
    id: 'sqlite',
    name: 'SQLiteCache',
    icon: <Database className="w-6 h-6" />,
    hitRate: 55,
    latency: 15,
    cost: 45,
    ttl: 86400,
    persistence: true,
    distributed: false,
    description: '持久化到 SQLite 数据库文件，重启后缓存依然有效',
    pros: ['持久化存储', '无需外部服务', '适合单机部署'],
    cons: ['无法分布式共享', '并发写入性能有限', '磁盘 I/O 开销'],
    bestFor: ['单机生产服务', '个人项目', '边缘计算节点']
  },
  {
    id: 'redis',
    name: 'RedisCache',
    icon: <Database className="w-6 h-6" />,
    hitRate: 70,
    latency: 8,
    cost: 30,
    ttl: 86400,
    persistence: true,
    distributed: true,
    description: '使用 Redis 作为分布式缓存，支持高并发与集群',
    pros: ['分布式共享', '高并发支持', '丰富的数据结构', 'TTL 自动过期'],
    cons: ['需要 Redis 服务', '网络延迟', '运维成本'],
    bestFor: ['分布式服务', '高并发 API', '多实例部署']
  },
  {
    id: 'semantic',
    name: 'SemanticCache',
    icon: <TrendingUp className="w-6 h-6" />,
    hitRate: 85,
    latency: 50,
    cost: 18,
    ttl: 86400,
    persistence: true,
    distributed: true,
    description: '基于语义相似度匹配，可容忍不同表述的相同问题',
    pros: ['命中率最高', '对用户表述鲁棒', '节省成本'],
    cons: ['需计算 embedding', '延迟略高', '可能返回不完全匹配结果'],
    bestFor: ['用户表述多样的问答系统', '客服机器人', 'FAQ 系统']
  }
];

export default function CachingStrategyComparison() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>('redis');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [requestCount, setRequestCount] = useState(100);

  const selected = strategies.find(s => s.id === selectedStrategy)!;

  const simulate = () => {
    setSimulationRunning(true);
    setTimeout(() => setSimulationRunning(false), 2000);
  };

  const calculateMetrics = (strategy: CachingStrategy) => {
    const hits = Math.floor(requestCount * (strategy.hitRate / 100));
    const misses = requestCount - hits;
    const avgLatency = (hits * strategy.latency + misses * 1200) / requestCount;
    const totalCost = (strategy.cost / 100) * misses;
    
    return { hits, misses, avgLatency, totalCost };
  };

  const metrics = calculateMetrics(selected);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">缓存策略对比</h3>
        <p className="text-gray-600">选择不同缓存策略，观察性能与成本差异</p>
      </div>

      {/* 策略选择器 */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        {strategies.map(strategy => (
          <motion.button
            key={strategy.id}
            onClick={() => setSelectedStrategy(strategy.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedStrategy === strategy.id
                ? 'border-blue-600 bg-blue-50'
                : 'border-gray-200 hover:border-blue-300'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="flex flex-col items-center gap-2">
              <div className={selectedStrategy === strategy.id ? 'text-blue-600' : 'text-gray-600'}>
                {strategy.icon}
              </div>
              <span className="text-sm font-medium text-center">{strategy.name}</span>
            </div>
          </motion.button>
        ))}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedStrategy}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 gap-6 mb-6"
      >
        {/* 左侧：描述与特性 */}
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">策略描述</h4>
            <p className="text-sm text-gray-700">{selected.description}</p>
          </div>

          <div>
            <h4 className="font-semibold mb-2 text-green-700">优点</h4>
            <ul className="text-sm space-y-1">
              {selected.pros.map((pro, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>{pro}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-2 text-red-700">缺点</h4>
            <ul className="text-sm space-y-1">
              {selected.cons.map((con, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-red-600 mt-0.5">✗</span>
                  <span>{con}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-2">最适场景</h4>
            <ul className="text-sm space-y-1">
              {selected.bestFor.map((scenario, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-blue-600 mt-0.5">→</span>
                  <span>{scenario}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* 右侧：性能指标 */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h4 className="font-semibold mb-4">核心指标</h4>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">缓存命中率</span>
                <span className="text-lg font-bold text-blue-600">{selected.hitRate}%</span>
              </div>
              <div className="w-full bg-gray-300 rounded-full h-2">
                <motion.div
                  className="bg-blue-600 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${selected.hitRate}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm font-medium flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  平均延迟
                </span>
                <span className="text-lg font-bold">{selected.latency}ms</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm font-medium flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  相对成本
                </span>
                <span className="text-lg font-bold">{selected.cost}%</span>
              </div>
              <div className="w-full bg-gray-300 rounded-full h-2">
                <motion.div
                  className="bg-green-600 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${100 - selected.cost}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-xs text-gray-600 mt-1">节省 {100 - selected.cost}% 成本</p>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-3 border-t">
              <div className="text-center">
                <div className="text-xs text-gray-600 mb-1">持久化</div>
                <div className={`text-sm font-semibold ${selected.persistence ? 'text-green-600' : 'text-red-600'}`}>
                  {selected.persistence ? '✓ 支持' : '✗ 不支持'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-600 mb-1">分布式</div>
                <div className={`text-sm font-semibold ${selected.distributed ? 'text-green-600' : 'text-red-600'}`}>
                  {selected.distributed ? '✓ 支持' : '✗ 不支持'}
                </div>
              </div>
            </div>

            <div className="pt-3 border-t">
              <div className="text-xs text-gray-600 mb-1">默认 TTL</div>
              <div className="text-sm font-semibold">
                {selected.ttl === 0 ? '不过期' : `${selected.ttl / 3600} 小时`}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 模拟器 */}
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-6">
        <h4 className="font-semibold mb-4">性能模拟器</h4>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            请求数量: {requestCount}
          </label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={requestCount}
            onChange={(e) => setRequestCount(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <button
          onClick={simulate}
          disabled={simulationRunning}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          {simulationRunning ? '模拟中...' : '运行模拟'}
        </button>

        {simulationRunning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 grid grid-cols-4 gap-3"
          >
            {[...Array(requestCount)].map((_, i) => (
              <motion.div
                key={i}
                className={`h-2 rounded ${
                  Math.random() * 100 < selected.hitRate ? 'bg-green-500' : 'bg-red-500'
                }`}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: i * (2 / requestCount), duration: 0.05 }}
              />
            ))}
          </motion.div>
        )}

        <div className="mt-6 grid grid-cols-4 gap-4">
          <div className="text-center p-3 bg-white rounded-lg">
            <div className="text-2xl font-bold text-green-600">{metrics.hits}</div>
            <div className="text-xs text-gray-600">缓存命中</div>
          </div>
          <div className="text-center p-3 bg-white rounded-lg">
            <div className="text-2xl font-bold text-red-600">{metrics.misses}</div>
            <div className="text-xs text-gray-600">缓存未命中</div>
          </div>
          <div className="text-center p-3 bg-white rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{metrics.avgLatency.toFixed(0)}ms</div>
            <div className="text-xs text-gray-600">平均延迟</div>
          </div>
          <div className="text-center p-3 bg-white rounded-lg">
            <div className="text-2xl font-bold text-purple-600">${metrics.totalCost.toFixed(2)}</div>
            <div className="text-xs text-gray-600">总成本</div>
          </div>
        </div>
      </div>
    </div>
  );
}

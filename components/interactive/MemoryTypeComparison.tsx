"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Zap, DollarSign, Target, TrendingUp } from 'lucide-react';

interface MemoryType {
  id: string;
  name: string;
  description: string;
  tokenCost: number;
  latency: number;
  contextQuality: number;
  complexity: number;
  useCase: string;
}

const MEMORY_TYPES: MemoryType[] = [
  {
    id: 'buffer',
    name: 'Buffer Memory',
    description: '保存完整对话历史',
    tokenCost: 100,
    latency: 10,
    contextQuality: 100,
    complexity: 20,
    useCase: '短对话（<10轮）、完整上下文需求'
  },
  {
    id: 'window',
    name: 'Window Memory',
    description: '滑动窗口（k=3）',
    tokenCost: 40,
    latency: 15,
    contextQuality: 60,
    complexity: 30,
    useCase: '实时聊天、性能敏感场景'
  },
  {
    id: 'summary',
    name: 'Summary Memory',
    description: '压缩为摘要',
    tokenCost: 25,
    latency: 50,
    contextQuality: 70,
    complexity: 60,
    useCase: '长对话（>50轮）、成本优化'
  },
  {
    id: 'summary_buffer',
    name: 'Summary+Buffer',
    description: '混合策略',
    tokenCost: 35,
    latency: 40,
    contextQuality: 85,
    complexity: 70,
    useCase: '平衡成本与质量'
  },
  {
    id: 'vector',
    name: 'Vector Memory',
    description: '语义检索',
    tokenCost: 30,
    latency: 80,
    contextQuality: 75,
    complexity: 90,
    useCase: '复杂查询、知识库对话'
  },
  {
    id: 'entity',
    name: 'Entity Memory',
    description: '实体跟踪',
    tokenCost: 45,
    latency: 60,
    contextQuality: 80,
    complexity: 85,
    useCase: '个性化服务、客户关系管理'
  }
];

const METRICS = [
  { key: 'tokenCost', label: 'Token 成本', icon: DollarSign, color: 'green', inverse: true },
  { key: 'latency', label: '延迟', icon: Zap, color: 'yellow', inverse: true },
  { key: 'contextQuality', label: '上下文质量', icon: Target, color: 'blue', inverse: false },
  { key: 'complexity', label: '实现复杂度', icon: TrendingUp, color: 'purple', inverse: true }
];

export default function MemoryTypeComparison() {
  const [selectedMetric, setSelectedMetric] = useState('tokenCost');
  const [comparisonMode, setComparisonMode] = useState<'chart' | 'table'>('chart');

  const getMetricColor = (value: number, inverse: boolean) => {
    const normalized = inverse ? 100 - value : value;
    if (normalized >= 75) return 'text-green-600';
    if (normalized >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getBarColor = (memoryId: string) => {
    const colors: Record<string, string> = {
      buffer: 'bg-blue-500',
      window: 'bg-green-500',
      summary: 'bg-purple-500',
      summary_buffer: 'bg-indigo-500',
      vector: 'bg-orange-500',
      entity: 'bg-pink-500'
    };
    return colors[memoryId] || 'bg-slate-500';
  };

  const maxValue = Math.max(...MEMORY_TYPES.map(m => m[selectedMetric as keyof MemoryType] as number));

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Memory Type Comparison</h3>
        <p className="text-slate-600">对比不同记忆类型的性能指标与适用场景</p>
      </div>

      {/* View Mode Toggle */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={() => setComparisonMode('chart')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              comparisonMode === 'chart'
                ? 'bg-blue-500 text-white'
                : 'bg-white text-slate-700 border border-slate-200'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            图表视图
          </button>
          <button
            onClick={() => setComparisonMode('table')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              comparisonMode === 'table'
                ? 'bg-blue-500 text-white'
                : 'bg-white text-slate-700 border border-slate-200'
            }`}
          >
            表格视图
          </button>
        </div>
      </div>

      {comparisonMode === 'chart' ? (
        <>
          {/* Metric Selection */}
          <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-3">
            {METRICS.map(metric => {
              const Icon = metric.icon;
              return (
                <button
                  key={metric.key}
                  onClick={() => setSelectedMetric(metric.key)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    selectedMetric === metric.key
                      ? 'bg-blue-50 border-blue-300 shadow-md'
                      : 'bg-white border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <Icon className={`w-5 h-5 mb-2 text-${metric.color}-500`} />
                  <div className="font-semibold text-sm">{metric.label}</div>
                </button>
              );
            })}
          </div>

          {/* Bar Chart */}
          <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
            <h4 className="font-semibold text-slate-800 mb-4">
              {METRICS.find(m => m.key === selectedMetric)?.label} 对比
              <span className="ml-2 text-xs text-slate-500">
                (值越{METRICS.find(m => m.key === selectedMetric)?.inverse ? '低' : '高'}越好)
              </span>
            </h4>
            
            <div className="space-y-4">
              {MEMORY_TYPES.map((memory, idx) => {
                const value = memory[selectedMetric as keyof MemoryType] as number;
                const percentage = (value / maxValue) * 100;
                
                return (
                  <motion.div
                    key={memory.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-40 text-sm font-medium text-slate-700">
                        {memory.name}
                      </div>
                      <div className="flex-1 bg-slate-100 rounded-full h-8 relative overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percentage}%` }}
                          transition={{ duration: 0.8, delay: idx * 0.1 }}
                          className={`h-full ${getBarColor(memory.id)} flex items-center justify-end pr-3`}
                        >
                          <span className="text-white text-sm font-semibold">
                            {value}
                          </span>
                        </motion.div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </>
      ) : (
        /* Table View */
        <div className="bg-white rounded-lg border border-slate-200 overflow-hidden mb-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                    记忆类型
                  </th>
                  {METRICS.map(metric => (
                    <th key={metric.key} className="px-4 py-3 text-center text-sm font-semibold text-slate-700">
                      {metric.label}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                    适用场景
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {MEMORY_TYPES.map(memory => (
                  <tr key={memory.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3">
                      <div className="font-medium text-slate-800">{memory.name}</div>
                      <div className="text-xs text-slate-500">{memory.description}</div>
                    </td>
                    {METRICS.map(metric => {
                      const value = memory[metric.key as keyof MemoryType] as number;
                      return (
                        <td key={metric.key} className="px-4 py-3 text-center">
                          <span className={`font-semibold ${getMetricColor(value, metric.inverse)}`}>
                            {value}
                          </span>
                        </td>
                      );
                    })}
                    <td className="px-4 py-3 text-sm text-slate-600">
                      {memory.useCase}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-green-50 rounded-lg border border-green-200 p-4">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-5 h-5 text-green-600" />
            <h4 className="font-semibold text-green-800">成本最优</h4>
          </div>
          <p className="text-sm text-green-700 mb-2">Summary Memory</p>
          <p className="text-xs text-green-600">适合长对话，极大节省 Token</p>
        </div>

        <div className="bg-blue-50 rounded-lg border border-blue-200 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-blue-600" />
            <h4 className="font-semibold text-blue-800">质量最优</h4>
          </div>
          <p className="text-sm text-blue-700 mb-2">Buffer Memory</p>
          <p className="text-xs text-blue-600">完整上下文，适合短对话</p>
        </div>

        <div className="bg-purple-50 rounded-lg border border-purple-200 p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-purple-600" />
            <h4 className="font-semibold text-purple-800">平衡推荐</h4>
          </div>
          <p className="text-sm text-purple-700 mb-2">Summary+Buffer</p>
          <p className="text-xs text-purple-600">混合策略，兼顾成本与质量</p>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">代码示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`# Buffer Memory - 完整历史
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# Window Memory - 滑动窗口
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=3)

# Summary Memory - 自动摘要
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# Summary+Buffer - 混合策略
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100
)

# Vector Memory - 语义检索
from langchain.memory import VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Entity Memory - 实体跟踪
from langchain.memory import ConversationEntityMemory
memory = ConversationEntityMemory(llm=llm)`}
        </pre>
      </div>
    </div>
  );
}

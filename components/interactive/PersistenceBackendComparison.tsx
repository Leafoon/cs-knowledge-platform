"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Zap, DollarSign, Shield, TrendingUp, Check, X } from 'lucide-react';

interface Backend {
  id: string;
  name: string;
  icon: typeof Database;
  description: string;
  performance: {
    writeLatency: number; // ms
    readLatency: number; // ms
    throughput: number; // req/s
  };
  features: {
    ttl: boolean;
    transactions: boolean;
    search: boolean;
    clustering: boolean;
  };
  cost: number; // 1-5 scale
  complexity: number; // 1-5 scale
  useCase: string;
  pros: string[];
  cons: string[];
}

const BACKENDS: Backend[] = [
  {
    id: 'redis',
    name: 'Redis',
    icon: Zap,
    description: '内存数据库，极速读写',
    performance: {
      writeLatency: 1,
      readLatency: 0.5,
      throughput: 100000
    },
    features: {
      ttl: true,
      transactions: true,
      search: false,
      clustering: true
    },
    cost: 3,
    complexity: 2,
    useCase: '高并发聊天应用、实时客服系统',
    pros: [
      '极低延迟（<1ms）',
      '支持 TTL 自动过期',
      '分布式集群支持',
      '丰富的数据结构'
    ],
    cons: [
      '内存限制',
      '数据持久化需配置',
      '成本较高（内存贵）'
    ]
  },
  {
    id: 'postgres',
    name: 'PostgreSQL',
    icon: Database,
    description: '关系型数据库，ACID 保证',
    performance: {
      writeLatency: 10,
      readLatency: 5,
      throughput: 10000
    },
    features: {
      ttl: false,
      transactions: true,
      search: true,
      clustering: true
    },
    cost: 2,
    complexity: 3,
    useCase: '企业级应用、需要复杂查询和事务',
    pros: [
      'ACID 事务保证',
      '强大的查询能力（JOIN、聚合）',
      '数据持久化可靠',
      'JSONB 支持灵活结构'
    ],
    cons: [
      '延迟较高（10ms+）',
      '写入性能受限',
      '水平扩展困难'
    ]
  },
  {
    id: 'mongodb',
    name: 'MongoDB',
    icon: Database,
    description: '文档数据库，灵活结构',
    performance: {
      writeLatency: 5,
      readLatency: 3,
      throughput: 50000
    },
    features: {
      ttl: true,
      transactions: true,
      search: true,
      clustering: true
    },
    cost: 2,
    complexity: 2,
    useCase: '快速迭代、非结构化数据、日志存储',
    pros: [
      '灵活的文档模型',
      '水平扩展能力强',
      '内置 TTL 索引',
      '丰富的查询语法'
    ],
    cons: [
      '事务能力较弱',
      '内存占用较大',
      'JOIN 操作复杂'
    ]
  },
  {
    id: 'file',
    name: 'File Storage',
    icon: Database,
    description: '本地文件存储',
    performance: {
      writeLatency: 50,
      readLatency: 20,
      throughput: 1000
    },
    features: {
      ttl: false,
      transactions: false,
      search: false,
      clustering: false
    },
    cost: 1,
    complexity: 1,
    useCase: '开发测试、单机部署、小规模应用',
    pros: [
      '零成本、零依赖',
      '简单易用',
      '便于调试和检查',
      '适合本地开发'
    ],
    cons: [
      '并发性能极差',
      '无事务保证',
      '无法分布式部署',
      '文件锁问题'
    ]
  }
];

const METRICS = [
  { key: 'writeLatency', label: '写入延迟', unit: 'ms', inverse: true },
  { key: 'readLatency', label: '读取延迟', unit: 'ms', inverse: true },
  { key: 'throughput', label: '吞吐量', unit: 'req/s', inverse: false },
  { key: 'cost', label: '成本', unit: '/5', inverse: true },
  { key: 'complexity', label: '复杂度', unit: '/5', inverse: true }
];

export default function PersistenceBackendComparison() {
  const [selectedBackend, setSelectedBackend] = useState<Backend>(BACKENDS[0]);
  const [comparisonMode, setComparisonMode] = useState<'detailed' | 'matrix'>('detailed');

  const getMetricValue = (backend: Backend, key: string): number => {
    if (key.includes('Latency') || key === 'throughput') {
      return backend.performance[key as keyof typeof backend.performance];
    }
    return backend[key as keyof Backend] as number;
  };

  const getPerformanceColor = (value: number, max: number, inverse: boolean) => {
    const ratio = inverse ? 1 - (value / max) : value / max;
    if (ratio >= 0.7) return 'text-green-600';
    if (ratio >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Persistence Backend Comparison
        </h3>
        <p className="text-slate-600">
          对比不同持久化后端的性能、成本与适用场景
        </p>
      </div>

      {/* View Toggle */}
      <div className="mb-6 flex gap-2">
        <button
          onClick={() => setComparisonMode('detailed')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            comparisonMode === 'detailed'
              ? 'bg-purple-500 text-white'
              : 'bg-white text-slate-700 border border-slate-200'
          }`}
        >
          详细对比
        </button>
        <button
          onClick={() => setComparisonMode('matrix')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            comparisonMode === 'matrix'
              ? 'bg-purple-500 text-white'
              : 'bg-white text-slate-700 border border-slate-200'
          }`}
        >
          矩阵视图
        </button>
      </div>

      {comparisonMode === 'detailed' ? (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Backend Selection */}
          <div>
            <h4 className="font-semibold text-slate-800 mb-3">选择存储后端</h4>
            <div className="space-y-3">
              {BACKENDS.map((backend) => {
                const Icon = backend.icon;
                return (
                  <button
                    key={backend.id}
                    onClick={() => setSelectedBackend(backend)}
                    className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                      selectedBackend.id === backend.id
                        ? 'border-purple-400 bg-purple-50 shadow-md'
                        : 'border-slate-200 bg-white hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${
                        selectedBackend.id === backend.id ? 'bg-purple-500' : 'bg-slate-200'
                      }`}>
                        <Icon className={`w-5 h-5 ${
                          selectedBackend.id === backend.id ? 'text-white' : 'text-slate-600'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-slate-800">
                          {backend.name}
                        </div>
                        <div className="text-xs text-slate-500">
                          {backend.description}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Backend Details */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h4 className="font-semibold text-slate-800 mb-4">
              {selectedBackend.name} 详情
            </h4>

            {/* Performance Metrics */}
            <div className="mb-6">
              <h5 className="text-sm font-semibold text-slate-700 mb-3">性能指标</h5>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">写入延迟</span>
                  <span className="font-semibold text-slate-800">
                    {selectedBackend.performance.writeLatency} ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">读取延迟</span>
                  <span className="font-semibold text-slate-800">
                    {selectedBackend.performance.readLatency} ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">吞吐量</span>
                  <span className="font-semibold text-slate-800">
                    {selectedBackend.performance.throughput.toLocaleString()} req/s
                  </span>
                </div>
              </div>
            </div>

            {/* Features */}
            <div className="mb-6">
              <h5 className="text-sm font-semibold text-slate-700 mb-3">功能特性</h5>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(selectedBackend.features).map(([key, supported]) => (
                  <div key={key} className="flex items-center gap-2">
                    {supported ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <X className="w-4 h-4 text-red-400" />
                    )}
                    <span className={`text-sm ${supported ? 'text-slate-700' : 'text-slate-400'}`}>
                      {key.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Pros & Cons */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div>
                <h5 className="text-sm font-semibold text-green-700 mb-2">优势</h5>
                <ul className="space-y-1">
                  {selectedBackend.pros.map((pro, idx) => (
                    <li key={idx} className="text-xs text-slate-600 flex items-start gap-1">
                      <span className="text-green-600 mt-0.5">✓</span>
                      <span>{pro}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h5 className="text-sm font-semibold text-red-700 mb-2">劣势</h5>
                <ul className="space-y-1">
                  {selectedBackend.cons.map((con, idx) => (
                    <li key={idx} className="text-xs text-slate-600 flex items-start gap-1">
                      <span className="text-red-600 mt-0.5">✗</span>
                      <span>{con}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Use Case */}
            <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-xs font-semibold text-purple-700 mb-1">
                适用场景
              </div>
              <div className="text-sm text-purple-900">
                {selectedBackend.useCase}
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Matrix View */
        <div className="bg-white rounded-lg border border-slate-200 overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                  后端
                </th>
                {METRICS.map(metric => (
                  <th key={metric.key} className="px-4 py-3 text-center text-sm font-semibold text-slate-700">
                    {metric.label}
                  </th>
                ))}
                <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700">
                  推荐场景
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {BACKENDS.map(backend => (
                <tr key={backend.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="font-medium text-slate-800">{backend.name}</div>
                    <div className="text-xs text-slate-500">{backend.description}</div>
                  </td>
                  {METRICS.map(metric => {
                    const value = getMetricValue(backend, metric.key);
                    const maxValue = Math.max(...BACKENDS.map(b => getMetricValue(b, metric.key)));
                    return (
                      <td key={metric.key} className="px-4 py-3 text-center">
                        <span className={`font-semibold ${getPerformanceColor(value, maxValue, metric.inverse)}`}>
                          {value.toLocaleString()}
                          <span className="text-xs ml-1 text-slate-400">{metric.unit}</span>
                        </span>
                      </td>
                    );
                  })}
                  <td className="px-4 py-3 text-sm text-slate-600">
                    {backend.useCase}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Code Examples */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-slate-900 text-slate-100 rounded-lg">
          <h4 className="font-semibold mb-3 text-sm">Redis 示例</h4>
          <pre className="text-xs font-mono overflow-x-auto">
{`from langchain_community.chat_message_histories import \\
    RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user_123",
    url="redis://localhost:6379/0",
    ttl=7 * 24 * 3600  # 7天过期
)`}
          </pre>
        </div>

        <div className="p-4 bg-slate-900 text-slate-100 rounded-lg">
          <h4 className="font-semibold mb-3 text-sm">PostgreSQL 示例</h4>
          <pre className="text-xs font-mono overflow-x-auto">
{`from langchain_community.chat_message_histories import \\
    PostgresChatMessageHistory

history = PostgresChatMessageHistory(
    connection_string="postgresql://...",
    session_id="sess_456"
)`}
          </pre>
        </div>
      </div>

      {/* Decision Guide */}
      <div className="mt-6 grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-green-50 rounded-lg border border-green-200">
          <Zap className="w-6 h-6 text-green-600 mb-2" />
          <h5 className="font-semibold text-green-800 mb-1">选择 Redis</h5>
          <p className="text-sm text-green-700">
            高并发、低延迟、实时应用
          </p>
        </div>

        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <Shield className="w-6 h-6 text-blue-600 mb-2" />
          <h5 className="font-semibold text-blue-800 mb-1">选择 PostgreSQL</h5>
          <p className="text-sm text-blue-700">
            事务保证、复杂查询、审计需求
          </p>
        </div>

        <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
          <TrendingUp className="w-6 h-6 text-purple-600 mb-2" />
          <h5 className="font-semibold text-purple-800 mb-1">选择 MongoDB</h5>
          <p className="text-sm text-purple-700">
            灵活结构、水平扩展、大规模数据
          </p>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Database, Layers, Search } from 'lucide-react';

interface PerformanceMetric {
  method: string;
  icon: React.ElementType;
  latency: number; // ms
  throughput: number; // queries/sec
  memory_mb: number;
  cache_hit_rate: number; // 0-100
  description: string;
  color: string;
}

const performanceData: PerformanceMetric[] = [
  {
    method: 'Vector Search (FAISS)',
    icon: Zap,
    latency: 15,
    throughput: 800,
    memory_mb: 250,
    cache_hit_rate: 0,
    description: '使用 FAISS 索引进行语义检索，精准度高但需要额外内存',
    color: 'blue'
  },
  {
    method: 'Cached Retrieval (LRU)',
    icon: Layers,
    latency: 2,
    throughput: 5000,
    memory_mb: 50,
    cache_hit_rate: 85,
    description: 'LRU 缓存热点查询，命中时极快，未命中时回退到原始检索',
    color: 'green'
  },
  {
    method: 'Paginated Loading',
    icon: Database,
    latency: 35,
    throughput: 300,
    memory_mb: 10,
    cache_hit_rate: 0,
    description: '分页加载大量历史消息，内存占用低但延迟较高',
    color: 'purple'
  },
  {
    method: 'Full Scan',
    icon: Search,
    latency: 120,
    throughput: 80,
    memory_mb: 500,
    cache_hit_rate: 0,
    description: '全量扫描所有消息，适用于小数据集或特殊场景',
    color: 'orange'
  }
];

export default function MemoryRetrievalPerformance() {
  const [selectedMethod, setSelectedMethod] = useState<string>('Vector Search (FAISS)');
  const [viewMode, setViewMode] = useState<'latency' | 'throughput' | 'memory'>('latency');

  const selected = performanceData.find(m => m.method === selectedMethod);

  const getColorClass = (color: string, variant: 'bg' | 'text' | 'border') => {
    const colors: Record<string, Record<string, string>> = {
      blue: { bg: 'bg-blue-500', text: 'text-blue-600', border: 'border-blue-500' },
      green: { bg: 'bg-green-500', text: 'text-green-600', border: 'border-green-500' },
      purple: { bg: 'bg-purple-500', text: 'text-purple-600', border: 'border-purple-500' },
      orange: { bg: 'bg-orange-500', text: 'text-orange-600', border: 'border-orange-500' }
    };
    return colors[color]?.[variant] || '';
  };

  const getMaxValue = (key: 'latency' | 'throughput' | 'memory_mb') => {
    return Math.max(...performanceData.map(m => m[key]));
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Memory Retrieval Performance Comparison
        </h3>
        <p className="text-slate-600">
          对比不同检索方法的性能指标
        </p>
      </div>

      {/* View Mode Toggle */}
      <div className="flex gap-2 mb-6">
        {(['latency', 'throughput', 'memory'] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === mode
                ? 'bg-blue-500 text-white'
                : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            {mode === 'latency' ? '延迟对比' : mode === 'throughput' ? '吞吐量对比' : '内存对比'}
          </button>
        ))}
      </div>

      {/* Performance Charts */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <h4 className="font-semibold text-slate-800 mb-4">
          {viewMode === 'latency' ? '查询延迟 (ms)' : 
           viewMode === 'throughput' ? '吞吐量 (queries/sec)' : 
           '内存占用 (MB)'}
        </h4>

        <div className="space-y-4">
          {performanceData.map((metric) => {
            const value = viewMode === 'latency' ? metric.latency :
                         viewMode === 'throughput' ? metric.throughput :
                         metric.memory_mb;
            const max = getMaxValue(viewMode === 'latency' ? 'latency' : 
                                    viewMode === 'throughput' ? 'throughput' : 
                                    'memory_mb');
            const percentage = (value / max) * 100;

            // For latency and memory, lower is better, so invert color
            const isBetter = viewMode === 'throughput' ? 
              percentage > 50 : percentage < 50;

            return (
              <div key={metric.method}>
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-2">
                    <metric.icon className={`w-5 h-5 ${getColorClass(metric.color, 'text')}`} />
                    <span className="font-medium text-slate-800">{metric.method}</span>
                  </div>
                  <span className="text-sm font-semibold text-slate-600">
                    {value.toLocaleString()}{viewMode === 'latency' ? ' ms' : viewMode === 'throughput' ? ' q/s' : ' MB'}
                  </span>
                </div>

                <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    className={getColorClass(metric.color, 'bg')}
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.8 }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Method Cards */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        {performanceData.map((metric) => (
          <motion.div
            key={metric.method}
            onClick={() => setSelectedMethod(metric.method)}
            className={`bg-white rounded-lg border-2 p-5 cursor-pointer transition-all ${
              selectedMethod === metric.method
                ? `${getColorClass(metric.color, 'border')} shadow-lg`
                : 'border-slate-200 hover:border-slate-300'
            }`}
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 rounded-lg ${getColorClass(metric.color, 'bg')} bg-opacity-10 flex items-center justify-center`}>
                <metric.icon className={`w-5 h-5 ${getColorClass(metric.color, 'text')}`} />
              </div>
              <h4 className="font-semibold text-slate-800">{metric.method}</h4>
            </div>

            <p className="text-sm text-slate-600 mb-4">{metric.description}</p>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-slate-500">延迟</div>
                <div className="font-semibold text-slate-800">{metric.latency} ms</div>
              </div>
              <div>
                <div className="text-xs text-slate-500">吞吐量</div>
                <div className="font-semibold text-slate-800">{metric.throughput} q/s</div>
              </div>
              <div>
                <div className="text-xs text-slate-500">内存</div>
                <div className="font-semibold text-slate-800">{metric.memory_mb} MB</div>
              </div>
              {metric.cache_hit_rate > 0 && (
                <div>
                  <div className="text-xs text-slate-500">缓存命中率</div>
                  <div className="font-semibold text-slate-800">{metric.cache_hit_rate}%</div>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detailed Metrics Table */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h4 className="font-semibold text-slate-800 mb-4">性能指标对比表</h4>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 text-sm font-semibold text-slate-700">检索方法</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">延迟 (ms)</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">吞吐量 (q/s)</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">内存 (MB)</th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-slate-700">缓存命中率</th>
              </tr>
            </thead>
            <tbody>
              {performanceData.map((metric) => (
                <tr key={metric.method} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <metric.icon className={`w-4 h-4 ${getColorClass(metric.color, 'text')}`} />
                      <span className="text-sm font-medium text-slate-800">{metric.method}</span>
                    </div>
                  </td>
                  <td className="text-right py-3 px-4 text-sm text-slate-600">
                    {metric.latency}
                  </td>
                  <td className="text-right py-3 px-4 text-sm text-slate-600">
                    {metric.throughput.toLocaleString()}
                  </td>
                  <td className="text-right py-3 px-4 text-sm text-slate-600">
                    {metric.memory_mb}
                  </td>
                  <td className="text-right py-3 px-4 text-sm text-slate-600">
                    {metric.cache_hit_rate > 0 ? `${metric.cache_hit_rate}%` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">性能测试代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`import time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Vector Search (FAISS)
def test_vector_search():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    start = time.time()
    results = vectorstore.similarity_search("query", k=5)
    latency = (time.time() - start) * 1000
    print(f"FAISS 延迟: {latency:.2f} ms")

# 2. Cached Retrieval (LRU)
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash: str):
    return memory.search(query)

start = time.time()
result = cached_retrieve(hash("query"))
latency = (time.time() - start) * 1000
print(f"缓存延迟: {latency:.2f} ms")

# 3. Paginated Loading
def test_pagination():
    page_size = 50
    total_pages = total_messages // page_size
    
    start = time.time()
    for page in range(total_pages):
        messages = memory.get_messages(
            offset=page * page_size,
            limit=page_size
        )
    latency = (time.time() - start) * 1000
    print(f"分页延迟: {latency:.2f} ms")`}
        </pre>
      </div>
    </div>
  );
}

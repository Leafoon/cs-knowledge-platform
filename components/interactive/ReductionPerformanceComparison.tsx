'use client';

import { useState } from 'react';

const methods = [
  {
    name: '串行归约',
    throughput: 50,
    latency: 1000,
    occupancy: '100%',
    useCase: '小规模数据',
  },
  {
    name: 'Warp Shuffle',
    throughput: 450,
    latency: 200,
    occupancy: '100%',
    useCase: '中等规模',
  },
  {
    name: '共享内存归约',
    throughput: 380,
    latency: 250,
    occupancy: '75%',
    useCase: '通用场景',
  },
  {
    name: '两阶段归约',
    throughput: 520,
    latency: 180,
    occupancy: '85%',
    useCase: '大规模数据',
  },
  {
    name: '向量化归约',
    throughput: 680,
    latency: 140,
    occupancy: '90%',
    useCase: '高带宽需求',
  },
];

export function ReductionPerformanceComparison() {
  const [selectedMethod, setSelectedMethod] = useState<number | null>(null);
  const maxThroughput = Math.max(...methods.map((m) => m.throughput));

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">归约性能对比</h2>
      
      <div className="space-y-3">
        {methods.map((m, i) => (
          <div
            key={m.name}
            onClick={() => setSelectedMethod(selectedMethod === i ? null : i)}
            className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
              selectedMethod === i ? 'border-blue-300 bg-blue-50' : 'border-transparent hover:bg-gray-50'
            }`}
          >
            <div className="flex justify-between items-center mb-2">
              <span className="font-medium text-gray-800">{m.name}</span>
              <span className="text-sm text-gray-500">{m.useCase}</span>
            </div>
            <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all"
                style={{ width: `${(m.throughput / maxThroughput) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>吞吐量: {m.throughput} GB/s</span>
              <span>延迟: {m.latency}ns</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-purple-50 rounded-lg text-sm text-purple-700">
        💡 向量化归约通过同时处理多个数据元素获得最高吞吐量
      </div>
    </div>
  );
}
'use client';

import { useState } from 'react';

const metrics = [
  { metric: '吞吐量 (tokens/s)', mha: '1,200', gqa: '2,400', mla: '3,600', best: 'mla' },
  { metric: 'KV Cache/Token', mha: '128B', gqa: '64B', mla: '32B', best: 'mla' },
  { metric: '内存 (32K seq)', mha: '16GB', gqa: '8GB', mla: '4GB', best: 'mla' },
  { metric: '训练 FLOPS', mha: '100%', gqa: '98%', mla: '95%', best: 'mha' },
  { metric: '注意力质量', mha: '100%', gqa: '97%', mla: '99%', best: 'mha' },
  { metric: '长序列扩展', mha: '差', gqa: '好', mla: '优', best: 'mla' },
];

const barData = [
  { name: 'MHA', throughput: 1200, memory: 16, color: '#EF4444' },
  { name: 'GQA', throughput: 2400, memory: 8, color: '#F59E0B' },
  { name: 'MLA', throughput: 3600, memory: 4, color: '#3B82F6' },
];

export default function MLAvsMHAvsGQAPerformance() {
  const [metric, setMetric] = useState<'throughput' | 'memory'>('throughput');

  const maxTP = 4000;
  const maxMem = 20;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">MHA vs GQA vs MLA 性能对比</h2>
      <p className="text-sm text-gray-400 mb-4">DeepSeek-V2 · 128K 上下文 · A100 推理</p>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setMetric('throughput')}
          className={`px-3 py-1 rounded text-sm ${metric === 'throughput' ? 'bg-blue-600' : 'bg-gray-700'}`}>
          吞吐量
        </button>
        <button onClick={() => setMetric('memory')}
          className={`px-3 py-1 rounded text-sm ${metric === 'memory' ? 'bg-blue-600' : 'bg-gray-700'}`}>
          内存占用
        </button>
      </div>

      {/* Bar chart */}
      <div className="flex items-end gap-6 justify-center mb-6 h-48">
        {barData.map((d, i) => {
          const val = metric === 'throughput' ? d.throughput : d.memory;
          const max = metric === 'throughput' ? maxTP : maxMem;
          const pct = (val / max) * 100;
          return (
            <div key={i} className="flex flex-col items-center" style={{ width: '100px' }}>
              <div className="text-sm font-bold mb-1" style={{ color: d.color }}>
                {metric === 'throughput' ? val.toLocaleString() : `${val}GB`}
              </div>
              <div className="w-full bg-gray-800 rounded-t-lg relative" style={{ height: '160px' }}>
                <div className="absolute bottom-0 w-full rounded-t-lg transition-all duration-500"
                  style={{ height: `${pct}%`, backgroundColor: d.color, opacity: 0.8 }} />
              </div>
              <div className="text-sm font-bold mt-2" style={{ color: d.color }}>{d.name}</div>
            </div>
          );
        })}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-2 py-1.5 text-left text-gray-400">指标</th>
              <th className="px-2 py-1.5 text-center text-red-400">MHA</th>
              <th className="px-2 py-1.5 text-center text-yellow-400">GQA</th>
              <th className="px-2 py-1.5 text-center text-blue-400">MLA</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="px-2 py-1.5 text-gray-300">{m.metric}</td>
                <td className="px-2 py-1.5 text-center" style={{ color: m.best === 'mha' ? '#EF4444' : '#9CA3AF' }}>
                  {m.mha} {m.best === 'mha' ? '⭐' : ''}
                </td>
                <td className="px-2 py-1.5 text-center" style={{ color: m.best === 'gqa' ? '#F59E0B' : '#9CA3AF' }}>
                  {m.gqa} {m.best === 'gqa' ? '⭐' : ''}
                </td>
                <td className="px-2 py-1.5 text-center" style={{ color: m.best === 'mla' ? '#3B82F6' : '#9CA3AF' }}>
                  {m.mla} {m.best === 'mla' ? '⭐' : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

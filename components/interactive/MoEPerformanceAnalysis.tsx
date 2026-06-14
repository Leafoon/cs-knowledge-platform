'use client';

import { useState } from 'react';

const models = [
  { name: 'Dense 7B', params: '7B', throughput: 100, memory: 14, latency: 45 },
  { name: 'MoE 16×7B', params: '112B (12B活跃)', throughput: 280, memory: 28, latency: 22 },
  { name: 'MoE 8×7B', params: '56B (7B活跃)', throughput: 195, memory: 18, latency: 32 },
  { name: 'MoE 32×7B', params: '224B (14B活跃)', throughput: 350, memory: 52, latency: 18 },
];

const metrics = [
  { key: 'throughput', label: '吞吐量 (tokens/s)', color: 'bg-blue-500' },
  { key: 'memory', label: '显存占用 (GB)', color: 'bg-red-500' },
  { key: 'latency', label: '延迟 (ms)', color: 'bg-green-500' },
];

export default function MoEPerformanceAnalysis() {
  const [selectedMetric, setSelectedMetric] = useState<string>('throughput');
  const [sortBy, setSortBy] = useState<'name' | 'throughput' | 'memory' | 'latency'>('throughput');

  const currentMetric = metrics.find(m => m.key === selectedMetric)!;
  const maxValue = Math.max(...models.map(m => m[selectedMetric as keyof typeof m] as number));

  const sortedModels = [...models].sort((a, b) => {
    const aVal = a[sortBy as keyof typeof a];
    const bVal = b[sortBy as keyof typeof b];
    if (typeof aVal === 'number' && typeof bVal === 'number') return bVal - aVal;
    return String(aVal).localeCompare(String(bVal));
  });

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">MoE 性能分析</h2>
      <p className="text-gray-400 text-sm mb-4">对比 MoE 模型与 Dense 模型的吞吐量和显存占用</p>

      <div className="flex gap-4 mb-6">
        <div className="flex gap-2">
          {metrics.map((m) => (
            <button
              key={m.key}
              onClick={() => setSelectedMetric(m.key)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                selectedMetric === m.key ? 'bg-blue-600' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
          className="bg-gray-800 text-gray-300 text-xs rounded px-2 py-1 border border-gray-600"
        >
          <option value="name">按名称</option>
          <option value="throughput">按吞吐量</option>
          <option value="memory">按显存</option>
          <option value="latency">按延迟</option>
        </select>
      </div>

      <div className="space-y-4 mb-6">
        {sortedModels.map((model) => {
          const value = model[selectedMetric as keyof typeof model] as number;
          const percentage = (value / maxValue) * 100;

          return (
            <div key={model.name} className="flex items-center gap-4">
              <div className="w-32 text-sm font-medium text-gray-300">{model.name}</div>
              <div className="flex-1 h-8 bg-gray-800 rounded overflow-hidden relative">
                <div
                  className={`h-full ${currentMetric.color} transition-all rounded`}
                  style={{ width: `${percentage}%` }}
                />
                <span className="absolute inset-y-0 right-2 flex items-center text-xs text-white font-mono">
                  {value}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold mb-3 text-gray-300">详细数据表</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-2">模型</th>
              <th className="text-right py-2">参数量</th>
              <th className="text-right py-2">吞吐量</th>
              <th className="text-right py-2">显存</th>
              <th className="text-right py-2">延迟</th>
              <th className="text-right py-2">效率提升</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.name} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                <td className="py-2 text-white">{model.name}</td>
                <td className="text-right text-gray-400">{model.params}</td>
                <td className="text-right text-blue-400">{model.throughput}</td>
                <td className="text-right text-red-400">{model.memory}GB</td>
                <td className="text-right text-green-400">{model.latency}ms</td>
                <td className="text-right text-yellow-400">
                  {model.name.includes('MoE')
                    ? `${((model.throughput / 100 - 1) * 100).toFixed(0)}%`
                    : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

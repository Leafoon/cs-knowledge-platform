'use client';

import { useState } from 'react';

const configs = [
  { name: '标准注意力', hbmAccess: '160GB', compute: '32 TFLOPS', flops: '12.4', memory: '16GB', latency: '高', color: '#EF4444' },
  { name: 'FlashAttention v1', hbmAccess: '40GB', compute: '28 TFLOPS', flops: '31.2', memory: '4GB', latency: '低', color: '#F59E0B' },
  { name: 'FlashAttention v2', hbmAccess: '35GB', compute: '30 TFLOPS', flops: '35.6', memory: '2GB', latency: '更低', color: '#3B82F6' },
  { name: 'PyTorch SDPA', hbmAccess: '42GB', compute: '26 TFLOPS', flops: '28.0', memory: '3.5GB', latency: '低', color: '#10B981' },
];

export default function FlashAttentionPerformanceComparison() {
  const [selectedMetric, setSelectedMetric] = useState<'hbmAccess' | 'memory' | 'flops'>('hbmAccess');

  const metricLabels = { hbmAccess: 'HBM 访问量', memory: '峰值内存', flops: '吞吐 TFLOPS' };
  const metricValues: Record<string, number[]> = {
    hbmAccess: [160, 40, 35, 42],
    memory: [16, 4, 2, 3.5],
    flops: [12.4, 31.2, 35.6, 28.0],
  };

  const maxVal = Math.max(...metricValues[selectedMetric]);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">FlashAttention 性能对比</h2>

      <div className="flex gap-2 mb-4">
        {(Object.keys(metricLabels) as Array<keyof typeof metricLabels>).map(m => (
          <button key={m} onClick={() => setSelectedMetric(m)}
            className={`px-3 py-1 rounded text-sm ${selectedMetric === m ? 'bg-blue-600' : 'bg-gray-700'}`}>
            {metricLabels[m]}
          </button>
        ))}
      </div>

      <div className="space-y-3 mb-6">
        {configs.map((c, i) => {
          const val = metricValues[selectedMetric][i];
          const isMax = selectedMetric === 'flops' ? val === maxVal : false;
          const isMin = selectedMetric !== 'flops' ? val === Math.min(...metricValues[selectedMetric]) : false;
          return (
            <div key={i} className="flex items-center gap-3">
              <span className="w-36 text-sm text-gray-300">{c.name}</span>
              <div className="flex-1 bg-gray-800 rounded-full h-6 relative">
                <div className="h-full rounded-full transition-all duration-500 flex items-center pr-2"
                  style={{
                    width: `${(val / maxVal) * 100}%`,
                    backgroundColor: c.color,
                    opacity: isMax || isMin ? 1 : 0.7,
                  }}>
                  <span className="text-xs font-bold text-white ml-auto">
                    {selectedMetric === 'hbmAccess' ? val + 'GB' : selectedMetric === 'memory' ? val + 'GB' : val}
                  </span>
                </div>
              </div>
              {selectedMetric === 'hbmAccess' && isMin && <span className="text-xs text-green-400 font-bold">最佳</span>}
              {selectedMetric === 'flops' && isMax && <span className="text-xs text-green-400 font-bold">最快</span>}
            </div>
          );
        })}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-2 py-1 text-left text-gray-400">方案</th>
              <th className="px-2 py-1 text-right text-gray-400">HBM (GB)</th>
              <th className="px-2 py-1 text-right text-gray-400">内存 (GB)</th>
              <th className="px-2 py-1 text-right text-gray-400">TFLOPS</th>
              <th className="px-2 py-1 text-right text-gray-400">延迟</th>
            </tr>
          </thead>
          <tbody>
            {configs.map((c, i) => (
              <tr key={i} className="border-b border-gray-800">
                <td className="px-2 py-1" style={{ color: c.color }}>{c.name}</td>
                <td className="px-2 py-1 text-right">{c.hbmAccess}</td>
                <td className="px-2 py-1 text-right">{c.memory}</td>
                <td className="px-2 py-1 text-right">{c.flops}</td>
                <td className="px-2 py-1 text-right">{c.latency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

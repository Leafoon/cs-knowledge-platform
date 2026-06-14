'use client';

import { useState } from 'react';

const benchmarkData = [
  { size: '512²', sparse: 120, dense: 85, sparsity: '50%' },
  { size: '1024²', sparse: 180, dense: 150, sparsity: '50%' },
  { size: '2048²', sparse: 240, dense: 220, sparsity: '50%' },
  { size: '4096²', sparse: 320, dense: 310, sparsity: '50%' },
  { size: '512²', sparse: 160, dense: 85, sparsity: '75%' },
  { size: '1024²', sparse: 250, dense: 150, sparsity: '75%' },
  { size: '2048²', sparse: 350, dense: 220, sparsity: '75%' },
  { size: '4096²', sparse: 450, dense: 310, sparsity: '75%' },
];

export default function SparseVsDensePerformance() {
  const [selectedSparsity, setSelectedSparsity] = useState<string>('50%');
  const [metric, setMetric] = useState<'throughput' | 'speedup'>('throughput');

  const filteredData = benchmarkData.filter(d => d.sparsity === selectedSparsity);
  const maxThroughput = Math.max(...benchmarkData.map(d => Math.max(d.sparse, d.dense)));

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">稀疏 vs 密集性能对比</h2>
      <p className="text-gray-400 text-sm mb-4">不同稀疏率下稀疏内核与密集内核的吞吐量对比</p>

      <div className="flex items-center gap-6 mb-6">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {['50%', '75%'].map((sp) => (
            <button
              key={sp}
              onClick={() => setSelectedSparsity(sp)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                selectedSparsity === sp ? 'bg-blue-600' : 'text-gray-400'
              }`}
            >
              {sp} 稀疏
            </button>
          ))}
        </div>
        <div className="flex bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setMetric('throughput')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
              metric === 'throughput' ? 'bg-green-600' : 'text-gray-400'
            }`}
          >
            吞吐量
          </button>
          <button
            onClick={() => setMetric('speedup')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
              metric === 'speedup' ? 'bg-green-600' : 'text-gray-400'
            }`}
          >
            加速比
          </button>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-end justify-between h-64 px-4">
          {filteredData.map((d, i) => {
            const sparseHeight = (d.sparse / maxThroughput) * 100;
            const denseHeight = (d.dense / maxThroughput) * 100;

            return (
              <div key={i} className="flex flex-col items-center gap-2 flex-1">
                <div className="flex gap-2 items-end h-56">
                  <div
                    className="w-10 bg-blue-500 rounded-t transition-all"
                    style={{ height: `${sparseHeight}%` }}
                    title={`稀疏: ${d.sparse} GFLOPS`}
                  />
                  <div
                    className="w-10 bg-gray-500 rounded-t transition-all"
                    style={{ height: `${denseHeight}%` }}
                    title={`密集: ${d.dense} GFLOPS`}
                  />
                </div>
                <div className="text-xs text-gray-400">{d.size}</div>
              </div>
            );
          })}
        </div>

        <div className="flex justify-center gap-6 mt-4">
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 bg-blue-500 rounded" />
            <span className="text-gray-400">稀疏内核</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 bg-gray-500 rounded" />
            <span className="text-gray-400">密集内核</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">性能提升详情</h3>
          <div className="space-y-3">
            {filteredData.map((d, i) => {
              const speedup = ((d.sparse / d.dense - 1) * 100).toFixed(1);
              return (
                <div key={i} className="flex items-center gap-4">
                  <div className="w-16 text-xs text-gray-400">{d.size}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-700 rounded-full">
                        <div
                          className="h-full bg-green-500 rounded-full"
                          style={{ width: `${Math.min(parseFloat(speedup) * 2, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-green-400 w-16 text-right">+{speedup}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">关键发现</h3>
          <div className="space-y-2 text-xs text-gray-400">
            <div className="flex items-start gap-2">
              <span className="text-blue-400 mt-0.5">●</span>
              <span>高稀疏率 (75%) 带来更显著的性能提升</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-400 mt-0.5">●</span>
              <span>大矩阵尺寸下稀疏优势更明显</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-yellow-400 mt-0.5">●</span>
              <span>TileLang 自动处理稀疏调度优化</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-purple-400 mt-0.5">●</span>
              <span>2:4 结构化稀疏有硬件原生支持</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

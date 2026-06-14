'use client';

import { useState } from 'react';

const benchmarks = [
  {
    name: 'GEMM (4096×4096×4096)',
    tileLang: 95,
    cuBLAS: 100,
    vendor: 92,
    cuda: 88,
  },
  {
    name: 'Flash Attention (seq=4096)',
    tileLang: 92,
    cuBLAS: 85,
    vendor: 90,
    cuda: 82,
  },
  {
    name: 'LayerNorm (batch=256)',
    tileLang: 85,
    cuBLAS: 78,
    vendor: 80,
    cuda: 75,
  },
  {
    name: 'Softmax (batch=256)',
    tileLang: 82,
    cuBLAS: 75,
    vendor: 78,
    cuda: 72,
  },
  {
    name: 'Conv2D (ResNet-50)',
    tileLang: 88,
    cuBLAS: 92,
    vendor: 87,
    cuda: 85,
  },
  {
    name: 'Transformer Block',
    tileLang: 90,
    cuBLAS: 86,
    vendor: 88,
    cuda: 80,
  },
];

export default function IndustrialKernelBenchmark() {
  const [selectedBenchmark, setSelectedBenchmark] = useState<number>(0);
  const [showNormalized, setShowNormalized] = useState(true);

  const current = benchmarks[selectedBenchmark];
  const implementations = [
    { name: 'TileLang', value: current.tileLang, color: 'bg-blue-500' },
    { name: 'cuBLAS', value: current.cuBLAS, color: 'bg-green-500' },
    { name: 'Vendor', value: current.vendor, color: 'bg-purple-500' },
    { name: 'CUDA', value: current.cuda, color: 'bg-orange-500' },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">工业级内核基准测试</h2>
      <p className="text-gray-400 text-sm mb-4">真实场景内核 vs cuBLAS vs 厂商实现的性能对比</p>

      <div className="flex items-center gap-4 mb-6">
        <select
          value={selectedBenchmark}
          onChange={(e) => setSelectedBenchmark(parseInt(e.target.value))}
          className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-gray-600"
        >
          {benchmarks.map((b, i) => (
            <option key={i} value={i}>{b.name}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-xs text-gray-400">
          <input
            type="checkbox"
            checked={showNormalized}
            onChange={(e) => setShowNormalized(e.target.checked)}
            className="rounded"
          />
          归一化显示
        </label>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-sm font-bold text-gray-300 mb-4">{current.name}</h3>
        <div className="space-y-4">
          {implementations.map((impl) => (
            <div key={impl.name} className="flex items-center gap-4">
              <div className="w-20 text-sm text-gray-400">{impl.name}</div>
              <div className="flex-1 h-8 bg-gray-700 rounded overflow-hidden relative">
                <div
                  className={`h-full ${impl.color} transition-all rounded`}
                  style={{
                    width: showNormalized
                      ? `${impl.value}%`
                      : `${(impl.value / 100) * 100}%`,
                  }}
                />
                <span className="absolute inset-y-0 right-2 flex items-center text-xs text-white font-bold">
                  {impl.value}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">各实现综合表现</h3>
          <div className="space-y-3">
            {['TileLang', 'cuBLAS', 'Vendor', 'CUDA'].map((name, idx) => {
              const avg = benchmarks.reduce((sum, b) => {
                const vals = [b.tileLang, b.cuBLAS, b.vendor, b.cuda];
                return sum + vals[idx];
              }, 0) / benchmarks.length;
              const colors = ['text-blue-400', 'text-green-400', 'text-purple-400', 'text-orange-400'];

              return (
                <div key={name}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className={colors[idx]}>{name}</span>
                    <span className="text-gray-300">{avg.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full">
                    <div
                      className={`h-full ${['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500'][idx]} rounded-full`}
                      style={{ width: `${avg}%` }}
                    />
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
              <span className="text-green-400 mt-0.5">✓</span>
              <span>TileLang 在大多数算子上接近或超越 cuBLAS</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-blue-400 mt-0.5">✓</span>
              <span>Flash Attention 场景下 TileLang 表现最优</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-purple-400 mt-0.5">✓</span>
              <span>厂商优化在特定硬件上有优势</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-yellow-400 mt-0.5">!</span>
              <span>纯 CUDA 手写内核性能仍有提升空间</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

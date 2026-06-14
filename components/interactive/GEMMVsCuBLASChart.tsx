'use client';

import { useState } from 'react';

const matrixSizes = [
  { m: 256, n: 256, k: 256 },
  { m: 512, n: 512, k: 512 },
  { m: 1024, n: 1024, k: 1024 },
  { m: 2048, n: 2048, k: 2048 },
  { m: 4096, n: 4096, k: 4096 },
  { m: 8192, n: 8192, k: 8192 },
];

const performanceData = [
  { size: '256³', tileLang: 85, cuBLAS: 80 },
  { size: '512³', tileLang: 88, cuBLAS: 82 },
  { size: '1024³', tileLang: 92, cuBLAS: 88 },
  { size: '2048³', tileLang: 94, cuBLAS: 92 },
  { size: '4096³', tileLang: 96, cuBLAS: 95 },
  { size: '8192³', tileLang: 97, cuBLAS: 98 },
];

export default function GEMMVsCuBLASChart() {
  const [selectedSize, setSelectedSize] = useState<number>(2);
  const [showTflops, setShowTflops] = useState(false);

  const maxPerf = 100;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">GEMM vs cuBLAS 性能对比</h2>
      <p className="text-gray-400 text-sm mb-4">不同矩阵尺寸下 TileLang GEMM 与 cuBLAS 的性能对比</p>

      <div className="flex items-center gap-4 mb-6">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {performanceData.map((d, i) => (
            <button
              key={i}
              onClick={() => setSelectedSize(i)}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                selectedSize === i ? 'bg-blue-600' : 'text-gray-400'
              }`}
            >
              {d.size}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-400 ml-auto">
          <input
            type="checkbox"
            checked={showTflops}
            onChange={(e) => setShowTflops(e.target.checked)}
            className="rounded"
          />
          显示 TFLOPS
        </label>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-end justify-between h-64 px-4">
          {performanceData.map((d, i) => {
            const tileLangHeight = (d.tileLang / maxPerf) * 100;
            const cuBLASHeight = (d.cuBLAS / maxPerf) * 100;
            const isSelected = i === selectedSize;

            return (
              <div
                key={i}
                className={`flex flex-col items-center gap-2 cursor-pointer transition-all ${
                  isSelected ? 'opacity-100' : 'opacity-70 hover:opacity-90'
                }`}
                onClick={() => setSelectedSize(i)}
              >
                <div className="flex gap-1 items-end h-56">
                  <div
                    className="w-8 bg-blue-500 rounded-t transition-all"
                    style={{ height: `${tileLangHeight}%` }}
                    title={`TileLang: ${d.tileLang}%`}
                  />
                  <div
                    className="w-8 bg-green-500 rounded-t transition-all"
                    style={{ height: `${cuBLASHeight}%` }}
                    title={`cuBLAS: ${d.cuBLAS}%`}
                  />
                </div>
                <div className={`text-xs ${isSelected ? 'text-white font-bold' : 'text-gray-400'}`}>
                  {d.size}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex justify-center gap-6 mt-4">
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 bg-blue-500 rounded" />
            <span className="text-gray-400">TileLang</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-3 h-3 bg-green-500 rounded" />
            <span className="text-gray-400">cuBLAS</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">当前选择: {performanceData[selectedSize].size}</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">TileLang</span>
              <span className="text-sm font-bold text-blue-400">{performanceData[selectedSize].tileLang}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">cuBLAS</span>
              <span className="text-sm font-bold text-green-400">{performanceData[selectedSize].cuBLAS}%</span>
            </div>
            <div className="flex items-center justify-between border-t border-gray-700 pt-2">
              <span className="text-xs text-gray-400">差距</span>
              <span className={`text-sm font-bold ${
                performanceData[selectedSize].tileLang >= performanceData[selectedSize].cuBLAS
                  ? 'text-green-400'
                  : 'text-red-400'
              }`}>
                {performanceData[selectedSize].tileLang >= performanceData[selectedSize].cuBLAS ? '+' : ''}
                {(performanceData[selectedSize].tileLang - performanceData[selectedSize].cuBLAS).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-gray-300 mb-3">性能分析</h3>
          <div className="space-y-2 text-xs text-gray-400">
            <div className="flex items-start gap-2">
              <span className="text-blue-400 mt-0.5">●</span>
              <span>小矩阵 (256-512): TileLang 调度优势明显</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-green-400 mt-0.5">●</span>
              <span>中矩阵 (1024-2048): 性能接近持平</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-yellow-400 mt-0.5">●</span>
              <span>大矩阵 (4096+): cuBLAS 内核优化更成熟</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

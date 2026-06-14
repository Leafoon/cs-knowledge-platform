'use client';

import { useState } from 'react';

const problemSizes = [
  { size: '32x32x32', achieved: 12.5, peak: 19.5, utilization: 64 },
  { size: '64x64x64', achieved: 45.2, peak: 19.5, utilization: 85 },
  { size: '128x128x128', achieved: 78.6, peak: 19.5, utilization: 92 },
  { size: '256x256x256', achieved: 85.3, peak: 19.5, utilization: 96 },
  { size: '512x512x512', achieved: 88.1, peak: 19.5, utilization: 98 },
];

export function TensorCoreUtilizationChart() {
  const [selectedSize, setSelectedSize] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Tensor Core 利用率分析</h2>
      
      <div className="mb-4 text-sm text-gray-500">
        峰值性能: 19.5 TFLOPS (FP16)
      </div>

      <div className="space-y-3">
        {problemSizes.map((p, i) => (
          <div
            key={p.size}
            onClick={() => setSelectedSize(selectedSize === i ? null : i)}
            className={`p-3 rounded-lg cursor-pointer transition-all ${
              selectedSize === i ? 'bg-purple-50 ring-2 ring-purple-300' : 'hover:bg-gray-50'
            }`}
          >
            <div className="flex justify-between text-sm mb-1">
              <span className="font-mono text-gray-600">{p.size}</span>
              <span className="text-gray-800">{p.achieved} / {p.peak} TFLOPS</span>
            </div>
            <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all"
                style={{ width: `${p.utilization}%` }}
              />
            </div>
            {selectedSize === i && (
              <div className="mt-2 flex justify-between text-sm">
                <span className="text-gray-500">利用率</span>
                <span className="font-bold text-purple-600">{p.utilization}%</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-purple-50 rounded-lg text-sm text-purple-700">
        💡 较大的问题规模能更好地利用Tensor Core的矩阵运算能力
      </div>
    </div>
  );
}
'use client';

import { useState } from 'react';

const patterns = [
  { name: 'Conv + ReLU', nonFused: 2.3, fused: 1.8, reduction: 22 },
  { name: 'Add + ReLU + Scale', nonFused: 1.5, fused: 0.9, reduction: 40 },
  { name: 'MatMul + Bias + GELU', nonFused: 4.2, fused: 3.1, reduction: 26 },
  { name: 'LayerNorm', nonFused: 1.8, fused: 1.2, reduction: 33 },
  { name: 'FlashAttention', nonFused: 8.5, fused: 5.2, reduction: 39 },
];

export function FusionPerformanceChart() {
  const [selectedPattern, setSelectedPattern] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">融合性能对比</h2>
      
      <div className="space-y-4">
        {patterns.map((p, i) => (
          <div
            key={p.name}
            onClick={() => setSelectedPattern(selectedPattern === i ? null : i)}
            className={`p-3 rounded-lg cursor-pointer transition-all ${
              selectedPattern === i ? 'bg-blue-50 ring-2 ring-blue-300' : 'hover:bg-gray-50'
            }`}
          >
            <div className="flex justify-between text-sm mb-2">
              <span className="font-medium text-gray-700">{p.name}</span>
              <span className="text-green-600 font-semibold">-{p.reduction}%</span>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-12">融合前</span>
                <div className="flex-1 h-4 bg-red-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-400"
                    style={{ width: `${(p.nonFused / 10) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 w-16 text-right">{p.nonFused}ms</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-12">融合后</span>
                <div className="flex-1 h-4 bg-green-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-400"
                    style={{ width: `${(p.fused / 10) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-600 w-16 text-right">{p.fused}ms</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
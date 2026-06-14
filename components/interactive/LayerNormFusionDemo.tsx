'use client';

import { useState } from 'react';

export function LayerNormFusionDemo() {
  const [showFused, setShowFused] = useState(false);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">LayerNorm 融合演示</h2>
      
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setShowFused(false)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            !showFused ? 'bg-red-500 text-white' : 'bg-gray-100'
          }`}
        >
          非融合实现
        </button>
        <button
          onClick={() => setShowFused(true)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            showFused ? 'bg-green-500 text-white' : 'bg-gray-100'
          }`}
        >
          融合实现
        </button>
      </div>

      {!showFused ? (
        <div className="space-y-3">
          {[
            { op: '计算均值 μ', io: '读X → 写μ', mem: '1次读 + 1次写' },
            { op: '计算方差 σ²', io: '读X, μ → 写σ²', mem: '1次读 + 1次写' },
            { op: '归一化', io: '读X, μ, σ² → 写X̂', mem: '1次读 + 1次写' },
            { op: '缩放和偏移', io: '读X̂, γ, β → 写Y', mem: '1次读 + 1次写' },
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-4 p-3 bg-red-50 rounded-lg">
              <div className="w-8 h-8 bg-red-200 rounded-full flex items-center justify-center text-red-700 font-bold">
                {i + 1}
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-800">{step.op}</div>
                <div className="text-sm text-gray-500">{step.io}</div>
              </div>
              <div className="text-sm text-red-600">{step.mem}</div>
            </div>
          ))}
          <div className="text-center text-sm text-red-600 mt-4">
            全局内存访问: 8次
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="p-4 bg-green-50 rounded-xl border-2 border-green-200">
            <div className="text-center mb-3">
              <span className="px-3 py-1 bg-green-200 rounded-full text-green-700 text-sm font-medium">
                单Pass融合内核
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center text-sm">
              <div className="p-2 bg-white rounded">
                <div className="text-green-500">读X</div>
              </div>
              <div className="p-2 bg-white rounded">
                <div className="text-green-500">μ + σ²</div>
              </div>
              <div className="p-2 bg-white rounded">
                <div className="text-green-500">X̂ = (X-μ)/σ</div>
              </div>
              <div className="p-2 bg-white rounded">
                <div className="text-green-500">Y = γX̂ + β</div>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-green-600 mt-4">
            全局内存访问: 2次 (读X, 写Y)
          </div>
        </div>
      )}

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 融合LayerNorm将内存访问减少4倍，显著提升性能
      </div>
    </div>
  );
}
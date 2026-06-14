'use client';

import { useState } from 'react';

export function OperatorFusionDiagram() {
  const [showFused, setShowFused] = useState(false);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">算子融合可视化</h2>
      
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setShowFused(false)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            !showFused ? 'bg-red-500 text-white' : 'bg-gray-100'
          }`}
        >
          融合前
        </button>
        <button
          onClick={() => setShowFused(true)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            showFused ? 'bg-green-500 text-white' : 'bg-gray-100'
          }`}
        >
          融合后
        </button>
      </div>

      {!showFused ? (
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-2">
            <div className="p-3 bg-blue-100 rounded-lg text-center">
              <div className="text-xs text-blue-500">输入</div>
              <div className="font-bold">X</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-3 bg-yellow-100 rounded-lg text-center">
              <div className="text-xs text-yellow-500">ReLU</div>
              <div className="font-bold">Y = max(0,X)</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-3 bg-blue-100 rounded-lg text-center">
              <div className="text-xs text-blue-500">中间结果</div>
              <div className="font-bold">Y</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-3 bg-purple-100 rounded-lg text-center">
              <div className="text-xs text-purple-500">Linear</div>
              <div className="font-bold">Z = WY + b</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-3 bg-blue-100 rounded-lg text-center">
              <div className="text-xs text-blue-500">输出</div>
              <div className="font-bold">Z</div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-500">2次全局内存写入</div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-2">
            <div className="p-3 bg-blue-100 rounded-lg text-center">
              <div className="text-xs text-blue-500">输入</div>
              <div className="font-bold">X</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-4 bg-green-100 rounded-lg text-center border-2 border-green-300">
              <div className="text-xs text-green-500">融合内核</div>
              <div className="font-bold text-sm">Z = W·max(0,X) + b</div>
            </div>
            <span className="text-2xl">→</span>
            <div className="p-3 bg-blue-100 rounded-lg text-center">
              <div className="text-xs text-blue-500">输出</div>
              <div className="font-bold">Z</div>
            </div>
          </div>
          <div className="text-center text-sm text-green-600 font-medium">仅1次全局内存写入 ✓</div>
        </div>
      )}
    </div>
  );
}
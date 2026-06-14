'use client';
import { useState } from 'react';

export function AscendAICoreMapping() {
  const [selectedCore, setSelectedCore] = useState<'cube' | 'vector' | null>(null);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">昇腾AI Core双引擎架构</h2>

      <div className="grid grid-cols-2 gap-6">
        {/* Cube Core */}
        <div onClick={() => setSelectedCore(selectedCore === 'cube' ? null : 'cube')}
          className={`border-2 rounded-xl p-5 cursor-pointer transition-all ${
            selectedCore === 'cube' ? 'border-cyan-400 bg-cyan-900/20' : 'border-gray-700 hover:border-gray-500'
          }`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-lg bg-cyan-700 flex items-center justify-center text-2xl">🧮</div>
            <div>
              <div className="font-bold text-cyan-300 text-lg">Cube Core</div>
              <div className="text-xs text-gray-400">矩阵计算引擎</div>
            </div>
          </div>

          <div className="space-y-2 text-sm">
            <div className="bg-gray-800 rounded p-2">
              <div className="text-cyan-400 text-xs font-medium">功能</div>
              <div className="text-gray-300">矩阵乘法 (MatMul)、卷积 (Conv)</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-cyan-400 text-xs font-medium">数据类型</div>
              <div className="text-gray-300">FP16, BF16, INT8</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-cyan-400 text-xs font-medium">计算规模</div>
              <div className="text-gray-300">16×16 矩阵单元</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-cyan-400 text-xs font-medium">峰值算力</div>
              <div className="text-gray-300">256 TFLOPS (BF16)</div>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-4 gap-1">
            {Array.from({ length: 16 }).map((_, i) => (
              <div key={i} className="aspect-square bg-cyan-800/60 rounded-sm" />
            ))}
          </div>
        </div>

        {/* Vector Core */}
        <div onClick={() => setSelectedCore(selectedCore === 'vector' ? null : 'vector')}
          className={`border-2 rounded-xl p-5 cursor-pointer transition-all ${
            selectedCore === 'vector' ? 'border-purple-400 bg-purple-900/20' : 'border-gray-700 hover:border-gray-500'
          }`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-lg bg-purple-700 flex items-center justify-center text-2xl">📐</div>
            <div>
              <div className="font-bold text-purple-300 text-lg">Vector Core</div>
              <div className="text-xs text-gray-400">向量计算引擎</div>
            </div>
          </div>

          <div className="space-y-2 text-sm">
            <div className="bg-gray-800 rounded p-2">
              <div className="text-purple-400 text-xs font-medium">功能</div>
              <div className="text-gray-300">ReLU, Softmax, Pooling, BN</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-purple-400 text-xs font-medium">数据类型</div>
              <div className="text-gray-300">FP16, BF16, FP32, INT32</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-purple-400 text-xs font-medium">向量宽度</div>
              <div className="text-gray-300">128-wide SIMD</div>
            </div>
            <div className="bg-gray-800 rounded p-2">
              <div className="text-purple-400 text-xs font-medium">峰值算力</div>
              <div className="text-gray-300">64 TFLOPS (FP16)</div>
            </div>
          </div>

          <div className="mt-3 flex gap-0.5">
            {Array.from({ length: 32 }).map((_, i) => (
              <div key={i} className="w-1.5 h-6 bg-purple-800/60 rounded-sm" />
            ))}
          </div>
        </div>
      </div>

      {/* Data Flow */}
      <div className="mt-6 border border-gray-700 rounded-xl p-4">
        <div className="text-sm text-gray-400 mb-3 font-medium">双引擎协同数据流</div>
        <div className="flex items-center justify-between text-xs">
          <div className="bg-blue-900/50 border border-blue-700 rounded-lg p-2 text-center">
            <div className="font-medium text-blue-300">L1 Cache</div>
            <div className="text-gray-400">输入数据</div>
          </div>
          <div className="text-gray-500">→</div>
          <div className="bg-cyan-900/50 border border-cyan-700 rounded-lg p-2 text-center">
            <div className="font-medium text-cyan-300">Cube Core</div>
            <div className="text-gray-400">矩阵计算</div>
          </div>
          <div className="text-gray-500">→</div>
          <div className="bg-purple-900/50 border border-purple-700 rounded-lg p-2 text-center">
            <div className="font-medium text-purple-300">Vector Core</div>
            <div className="text-gray-400">逐元素操作</div>
          </div>
          <div className="text-gray-500">→</div>
          <div className="bg-green-900/50 border border-green-700 rounded-lg p-2 text-center">
            <div className="font-medium text-green-300">UB Buffer</div>
            <div className="text-gray-400">输出结果</div>
          </div>
        </div>
      </div>

      {selectedCore && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <div className={`font-semibold ${selectedCore === 'cube' ? 'text-cyan-300' : 'text-purple-300'}`}>
            {selectedCore === 'cube' ? 'Cube Core' : 'Vector Core'} 详细信息
          </div>
          <div className="text-sm text-gray-300 mt-1">
            {selectedCore === 'cube'
              ? 'Cube Core是昇腾AI Core的核心矩阵计算单元，专为矩阵乘法和卷积运算优化。通过16×16的矩阵乘法单元，每周期可执行4096次乘加操作。支持与Vector Core的流水线协同。'
              : 'Vector Core负责逐元素运算和激活函数计算。支持128-wide的SIMD并行处理，可高效执行ReLU、Softmax、BatchNorm等操作。与Cube Core组成异构双引擎架构。'}
          </div>
        </div>
      )}
    </div>
  );
}

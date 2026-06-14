'use client';
import { useState } from 'react';

export function CubeVectorDualCoreDiagram() {
  const [selectedCore, setSelectedCore] = useState<'cube' | 'vector' | null>(null);
  const [animating, setAnimating] = useState(false);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">Cube + Vector 双引擎架构</h2>

      <div className="grid grid-cols-3 gap-6 items-start">
        {/* Cube Core */}
        <div onClick={() => setSelectedCore(selectedCore === 'cube' ? null : 'cube')}
          className={`border-2 rounded-xl p-5 cursor-pointer transition-all ${
            selectedCore === 'cube' ? 'border-cyan-400 bg-cyan-900/20 ring-2 ring-cyan-400/30' : 'border-gray-700 hover:border-gray-500'
          }`}>
          <div className="text-center mb-4">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-cyan-600 to-blue-600 flex items-center justify-center text-3xl font-bold mb-2">
              🧮
            </div>
            <div className="text-lg font-bold text-cyan-300">Cube Core</div>
            <div className="text-xs text-gray-400">矩阵计算引擎</div>
          </div>

          <div className="space-y-2">
            <div className="bg-gray-800 rounded-lg p-2">
              <div className="text-xs text-cyan-400 mb-1">计算单元</div>
              <div className="grid grid-cols-4 gap-0.5">
                {Array.from({ length: 16 }).map((_, i) => (
                  <div key={i} className="aspect-square bg-cyan-800/60 rounded-sm" />
                ))}
              </div>
            </div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>▸ 16×16 矩阵乘法单元</div>
              <div>▸ 支持 FP16/BF16/INT8</div>
              <div>▸ 峰值: 256 TFLOPS (BF16)</div>
              <div>▸ 输入: A[M×K] × B[K×N]</div>
            </div>
          </div>
        </div>

        {/* Data flow */}
        <div className="flex flex-col items-center justify-center gap-4 pt-16">
          <div className="border border-gray-700 rounded-lg p-3 bg-gray-800 text-center">
            <div className="text-xs text-gray-400 mb-1">UB Buffer</div>
            <div className="text-sm text-cyan-300">统一缓冲区</div>
          </div>
          <div className="flex gap-2">
            <svg className="w-8 h-8 text-cyan-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
          </div>
          <div className="border border-gray-700 rounded-lg p-3 bg-gray-800 text-center">
            <div className="text-xs text-gray-400 mb-1">Pipe</div>
            <div className="text-sm text-purple-300">流水线同步</div>
          </div>

          <button onClick={() => setAnimating(!animating)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">
            {animating ? '⏸ 暂停' : '▶ 播放数据流'}
          </button>

          {animating && (
            <div className="w-full h-8 relative overflow-hidden rounded bg-gray-800">
              <div className="absolute inset-0 flex items-center">
                <div className="w-4 h-4 bg-cyan-400 rounded-full animate-pulse" style={{
                  animation: 'slide 2s linear infinite',
                }} />
              </div>
              <style>{`@keyframes slide { 0% { transform: translateX(0); } 100% { transform: translateX(200px); } }`}</style>
            </div>
          )}
        </div>

        {/* Vector Core */}
        <div onClick={() => setSelectedCore(selectedCore === 'vector' ? null : 'vector')}
          className={`border-2 rounded-xl p-5 cursor-pointer transition-all ${
            selectedCore === 'vector' ? 'border-purple-400 bg-purple-900/20 ring-2 ring-purple-400/30' : 'border-gray-700 hover:border-gray-500'
          }`}>
          <div className="text-center mb-4">
            <div className="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center text-3xl font-bold mb-2">
              📐
            </div>
            <div className="text-lg font-bold text-purple-300">Vector Core</div>
            <div className="text-xs text-gray-400">向量计算引擎</div>
          </div>

          <div className="space-y-2">
            <div className="bg-gray-800 rounded-lg p-2">
              <div className="text-xs text-purple-400 mb-1">计算单元</div>
              <div className="flex gap-0.5">
                {Array.from({ length: 32 }).map((_, i) => (
                  <div key={i} className="w-1 h-4 bg-purple-800/60 rounded-sm" />
                ))}
              </div>
            </div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>▸ 128-wide SIMD向量单元</div>
              <div>▸ 支持 FP16/BF16/FP32</div>
              <div>▸ 峰值: 64 TFLOPS (FP16)</div>
              <div>▸ ReLU/Softmax/Pool/BN</div>
            </div>
          </div>
        </div>
      </div>

      {selectedCore && (
        <div className={`mt-4 p-4 rounded-lg border ${
          selectedCore === 'cube' ? 'bg-cyan-900/10 border-cyan-700' : 'bg-purple-900/10 border-purple-700'
        }`}>
          <div className={`font-semibold ${selectedCore === 'cube' ? 'text-cyan-300' : 'text-purple-300'}`}>
            {selectedCore === 'cube' ? 'Cube Core详解' : 'Vector Core详解'}
          </div>
          <div className="text-sm text-gray-300 mt-1">
            {selectedCore === 'cube'
              ? 'Cube Core负责矩阵乘法和卷积等计算密集型操作。通过16×16的矩阵乘法单元，每个时钟周期可执行4096次乘加运算。数据从UB Buffer通过Pipe流水线输入，计算结果写回UB Buffer供Vector Core使用。'
              : 'Vector Core负责逐元素操作和激活函数。支持128-wide的SIMD并行处理，可以高效执行ReLU、Softmax、BatchNorm等操作。与Cube Core通过UB Buffer进行数据交换，形成流水线协同。'}
          </div>
        </div>
      )}

      <div className="mt-4 p-3 bg-gray-800 rounded-lg text-xs text-gray-400">
        双引擎协同: Cube Core执行矩阵计算 → UB Buffer暂存 → Vector Core执行逐元素操作 → 输出结果
      </div>
    </div>
  );
}

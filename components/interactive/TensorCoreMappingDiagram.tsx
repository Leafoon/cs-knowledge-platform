'use client';
import { useState } from 'react';

const fragmentTypes = [
  { name: 'Matrix Fragment A', shape: '16×16', dtype: 'FP16', regs: 8, tilelang: 'Fragment<Matrix, A, 16, 16, 16>' },
  { name: 'Matrix Fragment B', shape: '16×16', dtype: 'FP16', regs: 8, tilelang: 'Fragment<Matrix, B, 16, 16, 16>' },
  { name: 'Matrix Fragment C', shape: '16×16', dtype: 'FP32', regs: 8, tilelang: 'Fragment<Matrix, C, 16, 16, 16>' },
  { name: 'Scalar Fragment', shape: '1×1', dtype: 'FP16', regs: 1, tilelang: 'Fragment<Scalar, A, 1, 1, 1>' },
];

const mmaInstructions = [
  { name: 'mma.sync.aligned.m16n8k16', threads: 32, shape: '16×8×16', compute: '2048 FLOP', latency: '~16 cycles' },
  { name: 'mma.sync.aligned.m8n8k4', threads: 32, shape: '8×8×4', compute: '512 FLOP', latency: '~8 cycles' },
  { name: 'mma.sync.aligned.m16n8k8', threads: 32, shape: '16×8×8', compute: '1024 FLOP', latency: '~12 cycles' },
];

export function TensorCoreMappingDiagram() {
  const [selectedFragment, setSelectedFragment] = useState(0);
  const [selectedMMA, setSelectedMMA] = useState(0);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">Tensor Core映射关系</h2>

      {/* Fragment types */}
      <div className="mb-6">
        <div className="text-sm text-gray-400 mb-3 font-medium">TileLang Fragment类型</div>
        <div className="grid grid-cols-4 gap-3">
          {fragmentTypes.map((f, i) => (
            <div key={i}
              onClick={() => setSelectedFragment(i)}
              className={`p-3 rounded-lg cursor-pointer transition-all border ${
                selectedFragment === i ? 'border-cyan-500 bg-cyan-900/20' : 'border-gray-700 hover:border-gray-500'
              }`}>
              <div className="text-sm font-medium text-cyan-300">{f.name}</div>
              <div className="text-xs text-gray-400 mt-1">Shape: {f.shape}</div>
              <div className="text-xs text-gray-400">Type: {f.dtype}</div>
              <div className="text-xs text-gray-400">寄存器: {f.regs}</div>
              <code className="text-[10px] text-green-400 mt-2 block break-all">{f.tilelang}</code>
            </div>
          ))}
        </div>
      </div>

      {/* Mapping flow */}
      <div className="mb-6 border border-gray-700 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-3 font-medium">Fragment → MMA指令 映射</div>
        <div className="flex items-center gap-4">
          <div className="bg-blue-900/50 border border-blue-700 rounded-lg p-3 text-center min-w-[120px]">
            <div className="text-blue-300 font-medium">{fragmentTypes[selectedFragment].name}</div>
            <div className="text-xs text-gray-400">{fragmentTypes[selectedFragment].shape}</div>
          </div>
          <div className="flex flex-col items-center text-gray-500">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <span className="text-[10px]">寄存器分配</span>
          </div>
          <div className="bg-green-900/50 border border-green-700 rounded-lg p-3 text-center min-w-[120px]">
            <div className="text-green-300 font-medium">Thread Regs</div>
            <div className="text-xs text-gray-400">每个线程 {fragmentTypes[selectedFragment].regs} 个寄存器</div>
          </div>
          <div className="flex flex-col items-center text-gray-500">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <span className="text-[10px]">Tensor Core</span>
          </div>
          <div className="bg-purple-900/50 border border-purple-700 rounded-lg p-3 text-center min-w-[120px]">
            <div className="text-purple-300 font-medium">{mmaInstructions[selectedMMA].name}</div>
            <div className="text-xs text-gray-400">{mmaInstructions[selectedMMA].shape}</div>
          </div>
        </div>
      </div>

      {/* MMA Instructions */}
      <div>
        <div className="text-sm text-gray-400 mb-3 font-medium">PTX MMA指令</div>
        <div className="space-y-2">
          {mmaInstructions.map((m, i) => (
            <div key={i}
              onClick={() => setSelectedMMA(i)}
              className={`p-3 rounded-lg cursor-pointer transition-all border ${
                selectedMMA === i ? 'border-green-500 bg-green-900/20' : 'border-gray-700 hover:border-gray-500'
              }`}>
              <div className="flex items-center justify-between">
                <code className="text-green-400 font-mono text-sm">{m.name}</code>
                <div className="flex gap-3 text-xs text-gray-400">
                  <span>线程: {m.threads}</span>
                  <span>计算: {m.compute}</span>
                  <span>延迟: {m.latency}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 p-3 bg-gray-800 rounded-lg text-xs text-gray-400">
        TileLang Fragment通过寄存器分配映射到32个线程，每个线程持有矩阵分块的一部分，最终通过mma.sync指令调用Tensor Core硬件执行矩阵乘加。
      </div>
    </div>
  );
}

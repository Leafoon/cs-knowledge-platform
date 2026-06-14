'use client';
import { useState } from 'react';

export function AMDWavefrontMapping() {
  const [hoveredLane, setHoveredLane] = useState<number | null>(null);
  const [selectedArch, setSelectedArch] = useState<'amd' | 'nvidia'>('amd');

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">AMD Wavefront vs NVIDIA Warp</h2>

      <div className="flex gap-3 mb-6">
        <button onClick={() => setSelectedArch('amd')}
          className={`px-4 py-2 rounded-lg text-sm transition-all ${selectedArch === 'amd' ? 'bg-red-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
          AMD Wavefront (64线程)
        </button>
        <button onClick={() => setSelectedArch('nvidia')}
          className={`px-4 py-2 rounded-lg text-sm transition-all ${selectedArch === 'nvidia' ? 'bg-green-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
          NVIDIA Warp (32线程)
        </button>
      </div>

      {selectedArch === 'amd' ? (
        <div>
          <div className="mb-4">
            <div className="text-sm text-gray-400 mb-2">AMD Wavefront — 64个线程以SIMT方式执行</div>
            <div className="grid grid-cols-8 gap-1">
              {Array.from({ length: 64 }).map((_, i) => (
                <div key={i}
                  className={`aspect-square rounded flex items-center justify-center text-[10px] font-mono cursor-pointer transition-all ${
                    hoveredLane === i ? 'bg-red-400 text-black scale-110' : 'bg-red-800 text-red-200'
                  }`}
                  onMouseEnter={() => setHoveredLane(i)}
                  onMouseLeave={() => setHoveredLane(null)}>
                  {i}
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="font-medium text-red-400 mb-2">Wavefront特性</div>
              <ul className="space-y-1 text-gray-300 text-xs">
                <li>▸ 64线程同时执行相同指令</li>
                <li>▸ 分为4个16-wide子wavefront</li>
                <li>▸ 每CU最多支持32个wavefront</li>
                <li>▸ Wavefront Size: 64</li>
                <li>▸ VGPR: 每线程最多256个</li>
              </ul>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="font-medium text-red-400 mb-2">MFMA指令</div>
              <ul className="space-y-1 text-gray-300 text-xs">
                <li>▸ MFMA_32x32x8F16</li>
                <li>▸ MFMA_16x16x16F16</li>
                <li>▸ MFMA_4x4x4F32</li>
                <li>▸ 每周期8次矩阵操作</li>
                <li>▸ 支持FP16/BF16/FP32</li>
              </ul>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div className="mb-4">
            <div className="text-sm text-gray-400 mb-2">NVIDIA Warp — 32个线程以SIMT方式执行</div>
            <div className="grid grid-cols-8 gap-1">
              {Array.from({ length: 32 }).map((_, i) => (
                <div key={i}
                  className={`aspect-square rounded flex items-center justify-center text-[10px] font-mono cursor-pointer transition-all ${
                    hoveredLane === i ? 'bg-green-400 text-black scale-110' : 'bg-green-800 text-green-200'
                  }`}
                  onMouseEnter={() => setHoveredLane(i)}
                  onMouseLeave={() => setHoveredLane(null)}>
                  {i}
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="font-medium text-green-400 mb-2">Warp特性</div>
              <ul className="space-y-1 text-gray-300 text-xs">
                <li>▸ 32线程同时执行相同指令</li>
                <li>▸ 每SM最多64个warp</li>
                <li>▸ 支持warp shuffle原语</li>
                <li>▸ Warp Size: 32</li>
                <li>▸ 每线程255个寄存器</li>
              </ul>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="font-medium text-green-400 mb-2">Tensor Core指令</div>
              <ul className="space-y-1 text-gray-300 text-xs">
                <li>▸ mma.sync PTX指令</li>
                <li>▸ ldmatrix共享内存加载</li>
                <li>▸ 16x16x16矩阵分块</li>
                <li>▸ 支持FP16/INT8/TF32</li>
                <li>▸ Warp级协作计算</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {hoveredLane !== null && (
        <div className="mt-4 p-2 bg-gray-800 rounded text-xs text-gray-400">
          Lane {hoveredLane} · {selectedArch === 'amd' ? 'Wavefront' : 'Warp'}成员
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-red-700" /> AMD MI300X
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-green-700" /> NVIDIA H100
        </div>
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';

export function KernelLaunchConfiguration() {
  const [blockDimX, setBlockDimX] = useState(256);
  const [gridDimX, setGridDimX] = useState(64);
  const [sharedMem, setSharedMem] = useState(48);

  const totalThreads = blockDimX * gridDimX;
  const warpsPerBlock = Math.ceil(blockDimX / 32);
  const registersPerBlock = warpsPerBlock * 32 * 32;
  const sharedMemPerBlock = sharedMem * 1024;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">内核启动配置</h2>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            blockDim.x: <span className="font-mono text-blue-600">{blockDimX}</span>
          </label>
          <input
            type="range"
            min="32"
            max="1024"
            step="32"
            value={blockDimX}
            onChange={(e) => setBlockDimX(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            gridDim.x: <span className="font-mono text-blue-600">{gridDimX}</span>
          </label>
          <input
            type="range"
            min="1"
            max="256"
            value={gridDimX}
            onChange={(e) => setGridDimX(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            共享内存: <span className="font-mono text-blue-600">{sharedMem}KB</span>
          </label>
          <input
            type="range"
            min="0"
            max="164"
            value={sharedMem}
            onChange={(e) => setSharedMem(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      <div className="p-4 bg-gray-50 rounded-xl mb-4">
        <div className="font-mono text-sm text-gray-700">
          kernel&lt;&lt;&lt;{gridDimX}, {blockDimX}, {sharedMem * 1024}&gt;&gt;&gt;(...)
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="p-3 bg-blue-50 rounded-lg text-center">
          <div className="text-sm text-blue-500">总线程数</div>
          <div className="text-lg font-bold text-blue-700">{totalThreads.toLocaleString()}</div>
        </div>
        <div className="p-3 bg-green-50 rounded-lg text-center">
          <div className="text-sm text-green-500">每块Warp数</div>
          <div className="text-lg font-bold text-green-700">{warpsPerBlock}</div>
        </div>
        <div className="p-3 bg-purple-50 rounded-lg text-center">
          <div className="text-sm text-purple-500">每块寄存器</div>
          <div className="text-lg font-bold text-purple-700">{registersPerBlock.toLocaleString()}</div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 线程块大小应为32的倍数，推荐使用128或256
      </div>
    </div>
  );
}
'use client';

import { useState } from 'react';

export function ThreadBlockScheduling() {
  const [blockSize, setBlockSize] = useState(256);
  const [gridSize, setGridSize] = useState(4);

  const warpsPerBlock = Math.ceil(blockSize / 32);
  const totalWarps = warpsPerBlock * gridSize;
  const maxWarpsPerSM = 48;
  const smNeeded = Math.ceil(totalWarps / maxWarpsPerSM);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">线程块调度</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            线程块大小: <span className="font-mono text-blue-600">{blockSize}</span>
          </label>
          <input
            type="range"
            min="32"
            max="1024"
            step="32"
            value={blockSize}
            onChange={(e) => setBlockSize(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            网格大小: <span className="font-mono text-blue-600">{gridSize}</span>
          </label>
          <input
            type="range"
            min="1"
            max="16"
            value={gridSize}
            onChange={(e) => setGridSize(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-center mb-6">
        <div className="p-3 bg-blue-50 rounded-lg">
          <div className="text-sm text-blue-500">每块Warp数</div>
          <div className="text-lg font-bold text-blue-700">{warpsPerBlock}</div>
        </div>
        <div className="p-3 bg-purple-50 rounded-lg">
          <div className="text-sm text-purple-500">总Warp数</div>
          <div className="text-lg font-bold text-purple-700">{totalWarps}</div>
        </div>
        <div className="p-3 bg-green-50 rounded-lg">
          <div className="text-sm text-green-500">需要SM数</div>
          <div className="text-lg font-bold text-green-700">{smNeeded}</div>
        </div>
      </div>

      <div className="p-4 bg-gray-50 rounded-xl">
        <div className="text-sm text-gray-500 mb-2">调度可视化</div>
        <div className="grid grid-cols-4 gap-2">
          {Array.from({ length: gridSize }, (_, i) => (
            <div key={i} className="p-2 bg-blue-100 rounded text-center text-xs">
              <div className="font-medium text-blue-700">Block {i}</div>
              <div className="text-blue-500">{blockSize} threads</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
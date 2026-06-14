'use client';

import { useState } from 'react';

export function RegisterAllocationStrategy() {
  const [registersPerThread, setRegistersPerThread] = useState(32);
  const [blockSize, setBlockSize] = useState(128);

  const maxRegisters = 65536;
  const threadsPerSM = Math.floor(maxRegisters / registersPerThread);
  const blocksPerSM = Math.floor(threadsPerSM / blockSize);
  const occupancy = Math.min((blocksPerSM * blockSize) / 2048, 1) * 100;
  const registerPressure = (registersPerThread / 255) * 100;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">寄存器分配策略</h2>
      
      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-600 mb-2">
              每线程寄存器数: <span className="font-mono text-blue-600">{registersPerThread}</span>
            </label>
            <input
              type="range"
              min="16"
              max="255"
              value={registersPerThread}
              onChange={(e) => setRegistersPerThread(Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
          
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
        </div>

        <div className="space-y-3">
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-500">SM可用寄存器</div>
            <div className="text-lg font-bold text-gray-800">{maxRegisters.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-500">每SM线程数</div>
            <div className="text-lg font-bold text-gray-800">{threadsPerSM.toLocaleString()}</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-500">每SM线程块数</div>
            <div className="text-lg font-bold text-gray-800">{blocksPerSM}</div>
          </div>
        </div>
      </div>

      <div className="mt-6">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600">占用率</span>
          <span className="font-mono text-gray-800">{occupancy.toFixed(1)}%</span>
        </div>
        <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${
              occupancy > 75 ? 'bg-green-500' : occupancy > 50 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${occupancy}%` }}
          />
        </div>
      </div>

      <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700">
        💡 权衡: 更多寄存器 → 更高指令级并行，但可能降低占用率
      </div>
    </div>
  );
}
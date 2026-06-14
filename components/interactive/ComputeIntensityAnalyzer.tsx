'use client';

import { useState } from 'react';

export function ComputeIntensityAnalyzer() {
  const [flops, setFlops] = useState(1000000);
  const [bytes, setBytes] = useState(1000000);

  const intensity = flops / bytes;
  const computeBound = intensity > 100;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">计算强度分析器</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm text-gray-600 mb-2">FLOPs (浮点操作数)</label>
          <input
            type="number"
            value={flops}
            onChange={(e) => setFlops(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-2">Bytes (字节数)</label>
          <input
            type="number"
            value={bytes}
            onChange={(e) => setBytes(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>
      </div>

      <div className="p-4 bg-gray-50 rounded-xl mb-4">
        <div className="text-center">
          <div className="text-sm text-gray-500 mb-1">计算强度 (FLOPs/Byte)</div>
          <div className="text-3xl font-bold text-gray-800">{intensity.toFixed(2)}</div>
        </div>
      </div>

      <div className="relative h-40 bg-gray-100 rounded-xl overflow-hidden mb-4">
        <div className="absolute inset-0 flex">
          <div className="w-1/2 bg-gradient-to-r from-blue-100 to-blue-50 flex items-center justify-center">
            <span className="text-sm text-blue-700 font-medium">内存密集型</span>
          </div>
          <div className="w-1/2 bg-gradient-to-r from-red-50 to-red-100 flex items-center justify-center">
            <span className="text-sm text-red-700 font-medium">计算密集型</span>
          </div>
        </div>
        <div
          className="absolute top-0 bottom-0 w-1 bg-black"
          style={{ left: `${Math.min(intensity / 2, 100)}%` }}
        />
        <div
          className="absolute w-4 h-4 bg-black rounded-full -mt-2 -ml-2"
          style={{ left: `${Math.min(intensity / 2, 100)}%`, top: '50%' }}
        />
      </div>

      <div className={`p-3 rounded-lg ${computeBound ? 'bg-red-50 text-red-700' : 'bg-blue-50 text-blue-700'}`}>
        💡 {computeBound ? '此内核是计算密集型，应优化计算效率' : '此内核是内存密集型，应优化内存访问'}
      </div>
    </div>
  );
}
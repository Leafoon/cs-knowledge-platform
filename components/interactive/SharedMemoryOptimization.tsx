'use client';

import { useState } from 'react';

const techniques = [
  {
    id: 'padding',
    name: '填充 (Padding)',
    desc: '添加额外列避免bank冲突',
    visual: (
      <div className="grid grid-cols-4 gap-1">
        {[...Array(16)].map((_, i) => (
          <div key={i} className={`w-10 h-10 rounded flex items-center justify-center text-xs ${
            i % 4 === 3 ? 'bg-gray-200 text-gray-400' : 'bg-blue-200 text-blue-700'
          }`}>
            {i % 4 === 3 ? 'P' : `A${i}`}
          </div>
        ))}
      </div>
    ),
  },
  {
    id: 'swizzling',
    name: '交错 (Swizzling)',
    desc: '重排访问模式消除冲突',
    visual: (
      <div className="grid grid-cols-4 gap-1">
        {[...Array(16)].map((_, i) => {
          const row = Math.floor(i / 4);
          const col = i % 4;
          const swizzled = (row * 4 + (col + row) % 4);
          return (
            <div key={i} className={`w-10 h-10 rounded flex items-center justify-center text-xs bg-purple-200 text-purple-700`}>
              {swizzled}
            </div>
          );
        })}
      </div>
    ),
  },
  {
    id: 'vectorized',
    name: '向量化加载',
    desc: '使用float4等宽类型一次加载多个元素',
    visual: (
      <div className="flex gap-2">
        {[0, 4, 8, 12].map((start) => (
          <div key={start} className="flex">
            {[0, 1, 2, 3].map((offset) => (
              <div key={offset} className="w-6 h-10 bg-green-200 border border-green-300 flex items-center justify-center text-[10px]">
                {start + offset}
              </div>
            ))}
            <span className="self-center text-xs text-green-600 ml-1">→</span>
          </div>
        ))}
      </div>
    ),
  },
];

export function SharedMemoryOptimization() {
  const [activeTechnique, setActiveTechnique] = useState('padding');

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">共享内存优化技术</h2>
      
      <div className="flex gap-2 mb-6">
        {techniques.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTechnique(t.id)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              activeTechnique === t.id ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      {techniques.map((t) => (
        activeTechnique === t.id && (
          <div key={t.id} className="space-y-4">
            <div className="p-4 bg-gray-50 rounded-xl">
              <div className="text-sm text-gray-500 mb-2">{t.desc}</div>
              <div className="flex justify-center py-4">{t.visual}</div>
            </div>
          </div>
        )
      ))}
    </div>
  );
}
'use client';

import { useState } from 'react';

export function MemoryCoalescingVisualization() {
  const [pattern, setPattern] = useState<'coalesced' | 'strided' | 'random'>('coalesced');

  const patterns = {
    coalesced: {
      title: '合并访问',
      accesses: [0, 1, 2, 3, 4, 5, 6, 7],
      transactions: 1,
      efficiency: '100%',
      desc: '连续线程访问连续地址',
    },
    strided: {
      title: '跨步访问',
      accesses: [0, 4, 8, 12, 16, 20, 24, 28],
      transactions: 8,
      efficiency: '12.5%',
      desc: '线程间隔访问',
    },
    random: {
      title: '随机访问',
      accesses: [7, 2, 15, 9, 4, 11, 0, 13],
      transactions: 8,
      efficiency: '12.5%',
      desc: '无规律访问模式',
    },
  };

  const current = patterns[pattern];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">内存访问模式可视化</h2>
      
      <div className="flex gap-2 mb-6">
        {(['coalesced', 'strided', 'random'] as const).map((p) => (
          <button
            key={p}
            onClick={() => setPattern(p)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              pattern === p ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {patterns[p].title}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-4 gap-1 mb-4">
        {Array.from({ length: 32 }, (_, i) => {
          const isAccessed = current.accesses.includes(i);
          return (
            <div
              key={i}
              className={`h-6 rounded text-xs flex items-center justify-center transition-all ${
                isAccessed
                  ? pattern === 'coalesced'
                    ? 'bg-green-400 text-white'
                    : 'bg-red-400 text-white'
                  : 'bg-gray-100 text-gray-400'
              }`}
            >
              {i}
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">内存事务</div>
          <div className="text-lg font-bold text-gray-800">{current.transactions}</div>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">带宽效率</div>
          <div className="text-lg font-bold text-gray-800">{current.efficiency}</div>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">模式</div>
          <div className="text-lg font-bold text-gray-800">{current.desc}</div>
        </div>
      </div>
    </div>
  );
}
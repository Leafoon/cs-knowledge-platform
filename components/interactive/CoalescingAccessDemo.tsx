'use client';

import { useState } from 'react';

export function CoalescingAccessDemo() {
  const [isCoalesced, setIsCoalesced] = useState(true);

  const coalesced = {
    title: '合并访问 (Coalesced)',
    transactions: 1,
    pattern: [0, 1, 2, 3, 4, 5, 6, 7],
    desc: '连续线程访问连续地址，单次事务完成',
  };

  const uncoalesced = {
    title: '非合并访问 (Uncoalesced)',
    transactions: 8,
    pattern: [0, 4, 1, 5, 2, 6, 3, 7],
    desc: '线程访问间隔地址，需要多次事务',
  };

  const current = isCoalesced ? coalesced : uncoalesced;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">内存合并访问演示</h2>
      
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setIsCoalesced(true)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            isCoalesced ? 'bg-green-500 text-white' : 'bg-gray-100'
          }`}
        >
          合并访问
        </button>
        <button
          onClick={() => setIsCoalesced(false)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            !isCoalesced ? 'bg-red-500 text-white' : 'bg-gray-100'
          }`}
        >
          非合并访问
        </button>
      </div>

      <div className="mb-6">
        <h3 className="font-semibold mb-3">{current.title}</h3>
        <div className="flex items-center gap-1 flex-wrap">
          {current.pattern.map((addr, i) => (
            <div
              key={i}
              className={`w-12 h-12 flex items-center justify-center rounded-lg border-2 ${
                isCoalesced
                  ? 'bg-green-100 border-green-300 text-green-700'
                  : 'bg-red-100 border-red-300 text-red-700'
              }`}
            >
              <div className="text-center">
                <div className="text-xs">T{i}</div>
                <div className="text-[10px]">→{addr}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-600">内存事务数:</span>
          <span className={`text-lg font-bold ${isCoalesced ? 'text-green-600' : 'text-red-600'}`}>
            {current.transactions}
          </span>
        </div>
        <div className="text-sm text-gray-600">{current.desc}</div>
      </div>
    </div>
  );
}
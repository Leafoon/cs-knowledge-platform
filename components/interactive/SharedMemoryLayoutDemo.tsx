'use client';

import React, { useState } from 'react';

type LayoutType = 'default' | 'padded' | 'swizzled';

interface BankInfo {
  id: number;
  threads: number[];
}

export function SharedMemoryLayoutDemo() {
  const [layout, setLayout] = useState<LayoutType>('default');
  const [showBanks, setShowBanks] = useState(true);

  const banks = 32;
  const threads = 32;

  const getBankId = (thread: number, type: LayoutType): number => {
    switch (type) {
      case 'padded':
        return thread % (banks + 4);
      case 'swizzled':
        return (thread ^ (thread >> 2)) % banks;
      default:
        return thread % banks;
    }
  };

  const getBankConflicts = (type: LayoutType): number => {
    const bankCounts = new Array(banks).fill(0);
    for (let t = 0; t < threads; t++) {
      const bank = getBankId(t, type);
      if (bank < banks) bankCounts[bank]++;
    }
    return bankCounts.filter((count) => count > 1).length;
  };

  const layouts = {
    default: {
      name: '默认布局',
      description: '线性分配，可能导致Bank Conflict',
      conflict: getBankConflicts('default'),
      code: `A_shared = T.alloc_shared((BM, BK), "float16")`,
    },
    padded: {
      name: 'Padding布局',
      description: '添加额外列避免冲突',
      conflict: getBankConflicts('padded'),
      code: `A_shared = T.alloc_shared((BM, BK + 4), "float16")`,
    },
    swizzled: {
      name: 'Swizzle布局',
      description: '使用XOR变换消除冲突',
      conflict: getBankConflicts('swizzled'),
      code: `A_shared = T.alloc_swizzle((BM, BK), "float16", swizzle_banks=32)`,
    },
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Shared Memory 布局演示</h2>
      
      {/* Layout selector */}
      <div className="flex gap-4 mb-6">
        {(Object.keys(layouts) as LayoutType[]).map((type) => (
          <button
            key={type}
            onClick={() => setLayout(type)}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              layout === type
                ? 'border-blue-500 bg-blue-900/30'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="text-white font-bold">{layouts[type].name}</div>
            <div className="text-gray-400 text-sm mt-1">{layouts[type].description}</div>
            <div className={`mt-2 text-sm ${layouts[type].conflict === 0 ? 'text-green-400' : 'text-red-400'}`}>
              冲突: {layouts[type].conflict} Banks
            </div>
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Bank visualization */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <span className="text-white font-bold">Bank 分配</span>
            <button
              onClick={() => setShowBanks(!showBanks)}
              className="text-blue-400 text-sm hover:underline"
            >
              {showBanks ? '隐藏' : '显示'} Bank ID
            </button>
          </div>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
            {Array.from({ length: threads }).map((_, t) => {
              const bank = getBankId(t, layout);
              return (
                <div
                  key={t}
                  className="w-6 h-6 flex items-center justify-center text-[10px] font-mono rounded"
                  style={{
                    backgroundColor: bank < banks && layouts[layout].conflict === 0
                      ? '#10B981'
                      : '#EF4444',
                  }}
                >
                  {showBanks ? bank : ''}
                </div>
              );
            })}
          </div>
          
          {/* Bank assignment table */}
          {showBanks && (
            <div className="mt-4 max-h-40 overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400">
                    <th className="text-left p-1">Thread</th>
                    <th className="text-left p-1">Bank</th>
                    <th className="text-left p-1">状态</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: 8 }).map((_, t) => {
                    const bank = getBankId(t, layout);
                    return (
                      <tr key={t} className="text-gray-300">
                        <td className="p-1">T{t}</td>
                        <td className="p-1">{bank}</td>
                        <td className="p-1">
                          {bank < banks ? (
                            <span className="text-green-400">✓</span>
                          ) : (
                            <span className="text-red-400">✗</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
        
        {/* Code and explanation */}
        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-white font-bold mb-2">代码实现</h3>
            <pre className="text-sm font-mono text-green-400 overflow-x-auto">
              {layouts[layout].code}
            </pre>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-white font-bold mb-2">性能影响</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Bank Conflicts:</span>
                <span className={layouts[layout].conflict === 0 ? 'text-green-400' : 'text-red-400'}>
                  {layouts[layout].conflict}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">有效带宽:</span>
                <span className="text-white">
                  {layouts[layout].conflict === 0 ? '100%' : `${Math.round(100 / (1 + layouts[layout].conflict * 0.1))}%`}
                </span>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-900/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">Bank Conflict 原理</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• Shared Memory 分为 32 个 Bank</li>
              <li>• 每个 Bank 宽度 4 字节</li>
              <li>• 同一 Bank 的并发访问串行化</li>
              <li>• 解决方案: Padding 或 Swizzle</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

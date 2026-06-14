'use client';

import React, { useState } from 'react';

export function SwizzledLayoutDemo() {
  const [showCode, setShowCode] = useState(false);
  const [hoveredBank, setHoveredBank] = useState<number | null>(null);

  const banks = 32;
  const threads = 32;

  const getOriginalBank = (thread: number) => thread % banks;
  const getSwizzledBank = (thread: number) => (thread ^ (thread >> 2)) % banks;

  const originalBanks = Array.from({ length: threads }, (_, i) => getOriginalBank(i));
  const swizzledBanks = Array.from({ length: threads }, (_, i) => getSwizzledBank(i));

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Swizzled Layout 演示</h2>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Before swizzle */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-red-400 font-bold mb-4">Before: 线性映射</h3>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
            {originalBanks.map((bank, i) => (
              <div
                key={i}
                className={`w-6 h-6 flex items-center justify-center text-[10px] font-mono rounded transition-all ${
                  hoveredBank === i ? 'ring-2 ring-white' : ''
                }`}
                style={{
                  backgroundColor: bank === i % banks ? '#EF4444' : '#374151',
                }}
                onMouseEnter={() => setHoveredBank(i)}
                onMouseLeave={() => setHoveredBank(null)}
              >
                {bank}
              </div>
            ))}
          </div>
          
          <div className="mt-4 text-gray-400 text-sm">
            Thread i → Bank (i % 32)
          </div>
          
          <div className="mt-2 text-red-400 text-sm">
            ⚠️ 存在 Bank Conflict
          </div>
        </div>
        
        {/* After swizzle */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-green-400 font-bold mb-4">After: XOR Swizzle</h3>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
            {swizzledBanks.map((bank, i) => (
              <div
                key={i}
                className={`w-6 h-6 flex items-center justify-center text-[10px] font-mono rounded transition-all ${
                  hoveredBank === i ? 'ring-2 ring-white' : ''
                }`}
                style={{
                  backgroundColor: '#10B981',
                }}
                onMouseEnter={() => setHoveredBank(i)}
                onMouseLeave={() => setHoveredBank(null)}
              >
                {bank}
              </div>
            ))}
          </div>
          
          <div className="mt-4 text-gray-400 text-sm">
            Thread i → Bank (i ^ (i &gt;&gt; 2)) % 32
          </div>
          
          <div className="mt-2 text-green-400 text-sm">
            ✓ 无 Bank Conflict
          </div>
        </div>
      </div>
      
      {/* XOR visualization */}
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-bold mb-4">XOR 操作可视化</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400">
                <th className="p-2 text-left">Thread</th>
                <th className="p-2 text-left">二进制</th>
                <th className="p-2 text-left">&gt;&gt; 2</th>
                <th className="p-2 text-left">XOR</th>
                <th className="p-2 text-left">% 32</th>
                <th className="p-2 text-left">Bank</th>
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3, 4, 5, 6, 7].map((t) => {
                const binary = t.toString(2).padStart(6, '0');
                const shifted = (t >> 2).toString(2).padStart(6, '0');
                const xored = (t ^ (t >> 2)).toString(2).padStart(6, '0');
                const bank = (t ^ (t >> 2)) % banks;
                return (
                  <tr key={t} className="text-gray-300 border-t border-gray-700">
                    <td className="p-2">{t}</td>
                    <td className="p-2 font-mono">{binary}</td>
                    <td className="p-2 font-mono">{shifted}</td>
                    <td className="p-2 font-mono">{xored}</td>
                    <td className="p-2">{bank}</td>
                    <td className="p-2">
                      <span className="px-2 py-1 rounded bg-green-900/50 text-green-400">
                        {bank}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Code */}
      <div className="mt-6">
        <button
          onClick={() => setShowCode(!showCode)}
          className="text-blue-400 hover:underline"
        >
          {showCode ? '收起代码' : '查看实现代码'}
        </button>
        
        {showCode && (
          <pre className="mt-4 bg-gray-800 rounded-lg p-4 text-sm font-mono text-green-400 overflow-x-auto">
{`# XOR Swizzle 实现
def swizzle_address(thread_id):
    # 线程ID右移2位后与原ID进行XOR
    return (thread_id ^ (thread_id >> 2)) % 32

# 在 TileLang 中
A_shared = T.alloc_swizzle(
    (BM, BK),
    "float16",
    swizzle_banks=32  # 32个Bank
)

# 优势:
# 1. 消除Bank Conflict
# 2. 无需额外Padding
# 3. 硬件友好的位操作`}
          </pre>
        )}
      </div>
    </div>
  );
}

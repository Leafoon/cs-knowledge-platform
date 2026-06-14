'use client';

import React, { useState } from 'react';

export function AllocSharedVisualizer() {
  const [showCode, setShowCode] = useState(true);
  const [selectedBank, setSelectedBank] = useState<number | null>(null);

  const banks = 32;
  const rows = 8;

  const getBankColor = (bank: number) => {
    if (selectedBank === null) return '#374151';
    return bank === selectedBank ? '#3B82F6' : '#1F2937';
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">T.alloc_shared 可视化</h2>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Code example */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-400 text-sm">代码示例</span>
            <button
              onClick={() => setShowCode(!showCode)}
              className="text-blue-400 text-sm hover:underline"
            >
              {showCode ? '收起' : '展开'}
            </button>
          </div>
          
          {showCode && (
            <pre className="text-sm font-mono text-green-400 overflow-x-auto">
{`# 分配 Shared Memory
A_shared = T.alloc_shared(
    (BM, BK),  # 形状: 128×32
    "float16"  # 数据类型
)

# 内存布局 (按行优先):
# Row 0: [0, 1, 2, ..., 31]
# Row 1: [32, 33, 34, ..., 63]
# ...

# 为了避免 Bank Conflict:
# 使用 Swizzle 或 Padding
A_shared = T.alloc_shared(
    (BM, BK + 4),  # +4 padding
    "float16"
)`}
            </pre>
          )}
        </div>
        
        {/* Memory layout */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-4">内存布局 (32 Banks)</h3>
          
          <div className="overflow-x-auto">
            <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
              {Array.from({ length: rows * banks }).map((_, i) => {
                const bank = i % banks;
                const row = Math.floor(i / banks);
                return (
                  <div
                    key={i}
                    className="w-6 h-6 flex items-center justify-center text-[10px] font-mono cursor-pointer transition-all"
                    style={{ backgroundColor: getBankColor(bank) }}
                    onClick={() => setSelectedBank(bank === selectedBank ? null : bank)}
                  >
                    {row === 0 ? bank : ''}
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* Bank legend */}
          <div className="mt-4 flex flex-wrap gap-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="flex items-center gap-1">
                <div
                  className="w-3 h-3 rounded cursor-pointer"
                  style={{ backgroundColor: getBankColor(i) }}
                  onClick={() => setSelectedBank(i === selectedBank ? null : i)}
                />
                <span className="text-gray-400 text-xs">Bank {i}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Annotations */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-green-900/30 rounded-lg p-4">
          <h4 className="text-green-400 font-bold mb-2">✅ 对齐优化</h4>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• 确保线程访问不同Bank</li>
            <li>• 使用Swizzle避免冲突</li>
            <li>• Padding增加4-8个元素</li>
          </ul>
        </div>
        
        <div className="bg-red-900/30 rounded-lg p-4">
          <h4 className="text-red-400 font-bold mb-2">❌ 常见错误</h4>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• 多线程访问同一Bank</li>
            <li>• 未对齐的内存访问</li>
            <li>• 过度使用Shared Memory</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

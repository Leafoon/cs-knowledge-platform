'use client';

import React, { useState } from 'react';

interface Approach {
  name: string;
  cycles: number;
  conflicts: number;
  color: string;
  description: string;
  code: string;
}

export function BankConflictComparison() {
  const [selectedApproach, setSelectedApproach] = useState<number>(0);

  const approaches: Approach[] = [
    {
      name: 'No Padding',
      cycles: 32,
      conflicts: 16,
      color: '#EF4444',
      description: '无任何优化，存在严重Bank Conflict',
      code: `# 无优化
A_shared = T.alloc_shared((BM, BK), "float16")
# Bank Conflict: 2-way
# 有效带宽: 50%`,
    },
    {
      name: 'Padding +4',
      cycles: 20,
      conflicts: 0,
      color: '#F59E0B',
      description: '添加4列Padding，消除Bank Conflict',
      code: `# Padding优化
A_shared = T.alloc_shared((BM, BK + 4), "float16")
# Bank Conflict: 0
# 有效带宽: 100%
# 额外开销: +4列 × BM × 2B`,
    },
    {
      name: 'Swizzle',
      cycles: 16,
      conflicts: 0,
      color: '#10B981',
      description: '使用XOR Swizzle，零冲突无额外开销',
      code: `# Swizzle优化
A_shared = T.alloc_swizzle(
    (BM, BK),
    "float16",
    swizzle_banks=32
)
# Bank Conflict: 0
# 有效带宽: 100%
# 额外开销: 0`,
    },
  ];

  const maxCycles = 40;

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Bank Conflict 对比</h2>
      
      {/* Comparison bars */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <h3 className="text-white font-bold mb-4">性能对比 (周期数)</h3>
        
        <div className="space-y-4">
          {approaches.map((approach, i) => (
            <div
              key={i}
              className={`p-4 rounded-lg cursor-pointer transition-all ${
                selectedApproach === i ? 'bg-gray-700' : 'hover:bg-gray-750'
              }`}
              onClick={() => setSelectedApproach(i)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-bold">{approach.name}</span>
                <span className="text-gray-400">{approach.cycles} cycles</span>
              </div>
              
              <div className="h-8 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all flex items-center justify-end pr-4"
                  style={{
                    width: `${(approach.cycles / maxCycles) * 100}%`,
                    backgroundColor: approach.color,
                  }}
                >
                  <span className="text-white font-bold text-sm">
                    {approach.cycles}
                  </span>
                </div>
              </div>
              
              <div className="mt-2 flex items-center gap-2">
                <span className="text-gray-400 text-sm">Conflicts:</span>
                <span className={`text-sm ${approach.conflicts === 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {approach.conflicts}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Detailed comparison table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden mb-6">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left p-3 text-gray-300">方法</th>
              <th className="text-right p-3 text-gray-300">周期数</th>
              <th className="text-right p-3 text-gray-300">Conflicts</th>
              <th className="text-right p-3 text-gray-300">带宽利用率</th>
              <th className="text-right p-3 text-gray-300">额外开销</th>
            </tr>
          </thead>
          <tbody>
            {approaches.map((approach, i) => (
              <tr
                key={i}
                className={`border-b border-gray-800 cursor-pointer transition-all ${
                  selectedApproach === i ? 'bg-gray-700' : ''
                }`}
                onClick={() => setSelectedApproach(i)}
              >
                <td className="p-3">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: approach.color }} />
                    <span className="text-white font-bold">{approach.name}</span>
                  </div>
                </td>
                <td className="p-3 text-right text-gray-300">{approach.cycles}</td>
                <td className={`p-3 text-right ${approach.conflicts === 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {approach.conflicts}
                </td>
                <td className={`p-3 text-right ${approach.conflicts === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {approach.conflicts === 0 ? '100%' : '50%'}
                </td>
                <td className="p-3 text-right text-gray-300">
                  {approach.name === 'Padding +4' ? '+4列' : '0'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Selected approach details */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: approaches[selectedApproach].color }} />
          <h3 className="text-white font-bold">{approaches[selectedApproach].name}</h3>
        </div>
        
        <p className="text-gray-300 mb-4">{approaches[selectedApproach].description}</p>
        
        <pre className="bg-gray-900 rounded-lg p-4 text-sm font-mono text-green-400 overflow-x-auto">
          {approaches[selectedApproach].code}
        </pre>
      </div>
    </div>
  );
}

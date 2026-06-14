'use client';

import { useState } from 'react';

const memoryLevels = [
  { level: '寄存器', size: '256KB/SM', bandwidth: '∞', latency: '0周期', color: 'bg-green-500', icon: '⚡' },
  { level: '共享内存', size: '48-164KB/SM', bandwidth: '19TB/s', latency: '20-30周期', color: 'bg-blue-500', icon: '💾' },
  { level: 'L2缓存', size: '40MB', bandwidth: '5TB/s', latency: '200周期', color: 'bg-purple-500', icon: '📦' },
  { level: '全局内存', size: '80GB', bandwidth: '2TB/s', latency: '400-600周期', color: 'bg-orange-500', icon: '🗄️' },
];

export function MemoryHierarchyPerformance() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">内存层级性能对比</h2>
      
      <div className="space-y-3">
        {memoryLevels.map((mem, index) => (
          <div
            key={mem.level}
            onClick={() => setSelectedLevel(selectedLevel === index ? null : index)}
            className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
              selectedLevel === index
                ? `${mem.color} bg-opacity-10 border-current`
                : 'border-transparent hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{mem.icon}</span>
              <div className="flex-1">
                <div className="font-semibold text-gray-800">{mem.level}</div>
                <div className="text-sm text-gray-500">容量: {mem.size}</div>
              </div>
              <div className="text-right">
                <div className="text-sm font-mono text-gray-700">带宽: {mem.bandwidth}</div>
                <div className="text-sm font-mono text-gray-700">延迟: {mem.latency}</div>
              </div>
            </div>
            {selectedLevel === index && (
              <div className="mt-3 pt-3 border-t border-gray-200 text-sm text-gray-600">
                {index === 0 && '最快但容量最小，编译器自动管理'}
                {index === 1 && '程序员显式管理，适合数据复用场景'}
                {index === 2 && '硬件自动管理，对程序员透明'}
                {index === 3 && '容量最大但延迟最高，需合并访问'}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 优化目标: 最大化数据复用，减少全局内存访问
      </div>
    </div>
  );
}
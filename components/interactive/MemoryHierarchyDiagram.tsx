'use client';

import React, { useState } from 'react';

interface MemoryLevel {
  name: string;
  color: string;
  size: string;
  bandwidth: string;
  latency: string;
  icon: string;
}

export function MemoryHierarchyDiagram() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);

  const levels: MemoryLevel[] = [
    { name: 'Registers', color: '#F59E0B', size: '256 KB/SM', bandwidth: '无限', latency: '~1 cycle', icon: '⚡' },
    { name: 'Shared Memory', color: '#10B981', size: '48-164 KB/SM', bandwidth: '19 TB/s', latency: '~20 cycles', icon: '📦' },
    { name: 'L2 Cache', color: '#8B5CF6', size: '40-80 MB', bandwidth: '6 TB/s', latency: '~100 cycles', icon: '💾' },
    { name: 'Global Memory', color: '#3B82F6', size: '16-80 GB', bandwidth: '2 TB/s', latency: '~400 cycles', icon: '🗄️' },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">GPU内存层次结构</h2>
      
      <div className="flex justify-center">
        <div className="relative">
          {/* Pyramid visualization */}
          <div className="flex flex-col items-center gap-0">
            {levels.map((level, i) => (
              <div
                key={i}
                className="relative cursor-pointer transition-all hover:scale-105"
                style={{
                  width: `${300 + i * 100}px`,
                  marginBottom: i < levels.length - 1 ? '-2px' : '0',
                }}
                onClick={() => setSelectedLevel(selectedLevel === i ? null : i)}
              >
                <div
                  className="p-4 rounded-lg border-2"
                  style={{
                    backgroundColor: selectedLevel === i ? level.color : `${level.color}40`,
                    borderColor: level.color,
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{level.icon}</span>
                      <span className="text-white font-bold">{level.name}</span>
                    </div>
                    <span className="text-gray-300 text-sm">{level.size}</span>
                  </div>
                  
                  {selectedLevel === i && (
                    <div className="mt-3 pt-3 border-t border-gray-600">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-400">带宽:</span>
                          <span className="text-white ml-2">{level.bandwidth}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">延迟:</span>
                          <span className="text-white ml-2">{level.latency}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          
          {/* Data flow arrows */}
          <div className="absolute right-0 top-0 bottom-0 w-16 flex flex-col items-center justify-center">
            <div className="text-gray-400 text-sm" style={{ writingMode: 'vertical-rl' }}>
              ↑ 速度递增 ↑
            </div>
          </div>
        </div>
      </div>
      
      {/* Bandwidth comparison */}
      <div className="mt-8 bg-gray-800 rounded-lg p-6">
        <h3 className="text-white font-bold mb-4">带宽对比</h3>
        <div className="space-y-3">
          {levels.map((level, i) => (
            <div key={i} className="flex items-center gap-4">
              <span className="w-32 text-gray-300 text-sm">{level.name}</span>
              <div className="flex-1 h-6 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${(i + 1) * 20}%`,
                    backgroundColor: level.color,
                  }}
                />
              </div>
              <span className="text-gray-400 text-sm w-20 text-right">{level.bandwidth}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-6 text-center text-gray-400 text-sm">
        点击各层查看详细信息 | 越靠近计算单元，速度越快
      </div>
    </div>
  );
}

'use client';

import React, { useState } from 'react';

interface MemoryLevel {
  name: string;
  color: string;
  capacity: string;
  latency: string;
  bandwidth: string;
  types: string[];
  description: string;
}

export function DataTypeHierarchy() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);

  const levels: MemoryLevel[] = [
    {
      name: 'Global Memory',
      color: '#3B82F6',
      capacity: '16-80 GB',
      latency: '~400 cycles',
      bandwidth: '2 TB/s',
      types: ['float32', 'float16', 'bfloat16', 'int8', 'int4'],
      description: 'GPU主存，所有线程可访问，延迟高但容量大',
    },
    {
      name: 'Shared Memory',
      color: '#10B981',
      capacity: '48-164 KB/SM',
      latency: '~20 cycles',
      bandwidth: '19 TB/s',
      types: ['float32', 'float16', 'int32', 'int16'],
      description: 'Block内共享，延迟低，需手动管理',
    },
    {
      name: 'Registers',
      color: '#F59E0B',
      capacity: '256 KB/SM',
      latency: '~1 cycle',
      bandwidth: '无限',
      types: ['float32', 'float16', 'int32', 'int8'],
      description: '最快存储，线程私有，容量最小',
    },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">数据类型层次结构</h2>
      
      <div className="flex flex-col items-center gap-4">
        {levels.map((level, i) => (
          <div
            key={i}
            className={`w-full max-w-2xl p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedLevel === i ? 'scale-105' : ''
            }`}
            style={{
              borderColor: level.color,
              backgroundColor: selectedLevel === i ? `${level.color}20` : '#1F2937',
            }}
            onClick={() => setSelectedLevel(selectedLevel === i ? null : i)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: level.color }} />
                <h3 className="text-white font-bold">{level.name}</h3>
              </div>
              <div className="text-gray-400 text-sm">{level.capacity}</div>
            </div>
            
            {selectedLevel === i && (
              <div className="mt-4 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-400 text-sm">延迟:</span>
                    <span className="text-white ml-2">{level.latency}</span>
                  </div>
                  <div>
                    <span className="text-gray-400 text-sm">带宽:</span>
                    <span className="text-white ml-2">{level.bandwidth}</span>
                  </div>
                </div>
                
                <div>
                  <span className="text-gray-400 text-sm">支持类型:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {level.types.map((type) => (
                      <span
                        key={type}
                        className="px-2 py-1 rounded text-xs font-mono"
                        style={{ backgroundColor: `${level.color}40`, color: level.color }}
                      >
                        {type}
                      </span>
                    ))}
                  </div>
                </div>
                
                <p className="text-gray-300 text-sm">{level.description}</p>
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Arrow indicators */}
      <div className="flex flex-col items-center gap-2 mt-4">
        <span className="text-gray-400 text-sm">↑ 容量递减，速度递增 ↑</span>
      </div>
      
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-bold mb-2">优化策略</h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• <strong>Global → Shared</strong>: 使用 Shared Memory 缓存频繁访问的数据</li>
          <li>• <strong>Shared → Registers</strong>: 使用寄存器存储计算中间结果</li>
          <li>• <strong>数据重用</strong>: 减少 Global Memory 访问次数</li>
          <li>• <strong>访存合并</strong>: 确保线程访问连续内存地址</li>
        </ul>
      </div>
    </div>
  );
}

'use client';
import { useState } from 'react';

const memoryLevels = [
  { name: '寄存器', size: '256KB/SM', bandwidth: '∞ (1周期)', color: 'bg-green-500', scope: 'Thread' },
  { name: '共享内存', size: '48KB/SM', bandwidth: '~30周期', color: 'bg-blue-500', scope: 'Block' },
  { name: 'L1缓存', size: '128KB/SM', bandwidth: '~30周期', color: 'bg-purple-500', scope: 'SM' },
  { name: 'L2缓存', size: '40MB', bandwidth: '~200周期', color: 'bg-orange-500', scope: 'Device' },
  { name: '全局内存 (HBM)', size: '80GB', bandwidth: '3.35TB/s', color: 'bg-red-500', scope: 'Device' },
];

export function CUDAThreadMapping() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);
  const [hoveredThread, setHoveredThread] = useState<number | null>(null);

  const gridThreads = 3;
  const blockThreads = 4;

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">CUDA线程层级映射</h2>

      {/* Thread Hierarchy */}
      <div className="flex items-start gap-8 mb-8">
        {/* Grid */}
        <div className="text-center">
          <div className="text-sm text-gray-400 mb-2">Grid</div>
          <div className="border-2 border-cyan-600 rounded-lg p-3 inline-block">
            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gridThreads}, 1fr)` }}>
              {Array.from({ length: gridThreads * gridThreads }).map((_, blockIdx) => (
                <div key={blockIdx}
                  className="border border-blue-500 rounded p-2 bg-blue-900/30"
                  style={{ width: 80 }}>
                  <div className="text-[10px] text-blue-300 mb-1">Block({blockIdx % gridThreads},{Math.floor(blockIdx / gridThreads)})</div>
                  <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(2, 1fr)` }}>
                    {Array.from({ length: Math.min(blockThreads, 4) }).map((_, threadIdx) => {
                      const globalId = blockIdx * blockThreads + threadIdx;
                      return (
                        <div key={threadIdx}
                          className={`text-[7px] text-center py-0.5 rounded ${
                            hoveredThread === globalId ? 'bg-green-500 text-black' : 'bg-gray-700'
                          }`}
                          onMouseEnter={() => setHoveredThread(globalId)}
                          onMouseLeave={() => setHoveredThread(null)}>
                          T{threadIdx}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Arrow */}
        <div className="flex items-center pt-16 text-gray-500">
          <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </div>

        {/* Memory Hierarchy */}
        <div className="flex-1">
          <div className="text-sm text-gray-400 mb-2">内存层级</div>
          <div className="space-y-2">
            {memoryLevels.map((level, i) => (
              <div key={i}
                onClick={() => setSelectedLevel(selectedLevel === i ? null : i)}
                className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${
                  selectedLevel === i ? 'bg-gray-800 ring-1 ring-cyan-500' : 'hover:bg-gray-800/50'
                }`}>
                <div className={`w-4 h-4 rounded-full ${level.color}`} />
                <div className="flex-1">
                  <div className="text-sm font-medium">{level.name}</div>
                  <div className="text-xs text-gray-400">{level.size} · {level.bandwidth}</div>
                </div>
                <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">{level.scope}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {selectedLevel !== null && (
        <div className="p-4 bg-gray-800 rounded-lg">
          <div className="font-semibold text-cyan-300">{memoryLevels[selectedLevel].name}</div>
          <div className="text-sm text-gray-300 mt-1">
            {selectedLevel === 0 && '每个线程独享的寄存器，访问速度最快。受寄存器压力限制。'}
            {selectedLevel === 1 && 'Block内线程共享，可用于线程间通信和数据复用。'}
            {selectedLevel === 2 && '每个SM独享的L1缓存，自动缓存全局内存访问。'}
            {selectedLevel === 3 && '所有SM共享，容量较大但延迟较高。'}
            {selectedLevel === 4 && '高带宽HBM显存，延迟最高但容量最大。'}
          </div>
        </div>
      )}

      {hoveredThread !== null && (
        <div className="mt-4 p-2 bg-green-900/30 rounded text-xs text-green-300">
          当前线程: {hoveredThread} (Block内局部ID: {hoveredThread % blockThreads})
        </div>
      )}
    </div>
  );
}

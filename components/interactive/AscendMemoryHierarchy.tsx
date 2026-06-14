'use client';
import { useState } from 'react';

const memoryLevels = [
  {
    name: 'UB (Unified Buffer)',
    fullName: '统一缓冲区',
    size: '192 KB/Core',
    bandwidth: '最高 (1周期)',
    color: 'bg-green-500',
    border: 'border-green-500',
    textColor: 'text-green-300',
    scope: 'AI Core',
    description: 'AI Core内部的统一缓冲区，Cube和Vector Core共享访问',
    details: ['直接连接Cube/Vector计算单元', '支持读写操作', '用于中间结果暂存', '通过Pipe与L1连接'],
  },
  {
    name: 'L1 Buffer',
    fullName: '一级缓存',
    size: '256 KB/Tile',
    bandwidth: '~4周期',
    color: 'bg-blue-500',
    border: 'border-blue-500',
    textColor: 'text-blue-300',
    scope: 'Tile',
    description: 'Tile级缓存，供多个AI Core共享',
    details: ['Tile内AI Core共享', '缓存GM数据', '支持数据预取', 'L1→UB数据搬运'],
  },
  {
    name: 'L2 Cache',
    fullName: '二级缓存',
    size: '4 MB',
    bandwidth: '~10周期',
    color: 'bg-purple-500',
    border: 'border-purple-500',
    textColor: 'text-purple-300',
    scope: 'L2 Cache',
    description: '所有AI Core共享的二级缓存',
    details: ['全局共享', '大容量缓存', '减少HBM访问', '硬件预取支持'],
  },
  {
    name: 'GM (Global Memory)',
    fullName: '全局内存 (HBM)',
    size: '64 GB HBM',
    bandwidth: '1.6 TB/s',
    color: 'bg-red-500',
    border: 'border-red-500',
    textColor: 'text-red-300',
    scope: 'Device',
    description: '高带宽显存，容量最大但延迟最高',
    details: ['64GB HBM2e', '1.6TB/s带宽', 'CPU/GPU共享', 'DMA搬运'],
  },
];

export function AscendMemoryHierarchy() {
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null);

  const bandwidthScale = [100, 60, 30, 10];

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-orange-400 mb-6">昇腾内存层级架构</h2>

      {/* Memory hierarchy visualization */}
      <div className="flex flex-col items-center mb-6">
        {memoryLevels.map((level, i) => (
          <div key={i} className="flex flex-col items-center">
            <div
              onClick={() => setSelectedLevel(selectedLevel === i ? null : i)}
              className={`border-2 ${level.border} rounded-xl p-4 cursor-pointer transition-all ${
                selectedLevel === i ? 'bg-gray-800 ring-2 ring-white/20 scale-105' : 'hover:bg-gray-800/50'
              }`}
              style={{ width: `${300 + i * 100}px` }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-4 h-4 rounded-full ${level.color}`} />
                  <div>
                    <div className={`font-bold text-sm ${level.textColor}`}>{level.name}</div>
                    <div className="text-xs text-gray-400">{level.fullName}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-mono text-gray-300">{level.size}</div>
                  <div className="text-xs text-gray-500">{level.bandwidth}</div>
                </div>
              </div>
            </div>
            {i < memoryLevels.length - 1 && (
              <div className="flex flex-col items-center py-1">
                <div className="w-px h-4 bg-gray-600" />
                <div className="text-[10px] text-gray-500">↕</div>
                <div className="w-px h-4 bg-gray-600" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Bandwidth comparison */}
      <div className="mb-6 border border-gray-700 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-3">带宽对比 (相对值)</div>
        <div className="space-y-2">
          {memoryLevels.map((level, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className={`w-24 text-xs ${level.textColor}`}>{level.name.split('(')[0]}</span>
              <div className="flex-1 bg-gray-800 rounded-full h-4 overflow-hidden">
                <div className={`h-full ${level.color} rounded-full transition-all`}
                  style={{ width: `${bandwidthScale[i]}%` }} />
              </div>
              <span className="w-16 text-xs text-gray-400 text-right">{level.bandwidth}</span>
            </div>
          ))}
        </div>
      </div>

      {selectedLevel !== null && (
        <div className={`p-4 rounded-lg border ${memoryLevels[selectedLevel].border} bg-gray-800/50`}>
          <div className={`font-semibold ${memoryLevels[selectedLevel].textColor}`}>
            {memoryLevels[selectedLevel].name} — {memoryLevels[selectedLevel].fullName}
          </div>
          <div className="text-sm text-gray-300 mt-1">{memoryLevels[selectedLevel].description}</div>
          <div className="grid grid-cols-2 gap-2 mt-3">
            {memoryLevels[selectedLevel].details.map((d, j) => (
              <div key={j} className="bg-gray-800 rounded px-2 py-1 text-xs text-gray-400 flex items-center gap-1">
                <span className="text-orange-500">▸</span>{d}
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-gray-500">访问范围: {memoryLevels[selectedLevel].scope}</div>
        </div>
      )}
    </div>
  );
}

'use client';

import { useState, useMemo } from 'react';

function calcRegisters(tileSize: number, dtype: 'fp16' | 'fp32' = 'fp16'): number {
  const bytesPerElem = dtype === 'fp16' ? 2 : 4;
  const smemPerTile = tileSize * tileSize * bytesPerElem;
  const regA = tileSize * tileSize;
  const regB = tileSize * tileSize;
  const regC = tileSize * tileSize;
  const overhead = 32;
  return regA + regB + regC + overhead;
}

function calcOccupancy(registers: number, maxRegisters = 65536, threadsPerSM = 2048): number {
  const regPerThread = Math.ceil(registers);
  const maxThreads = Math.floor(maxRegisters / regPerThread);
  return Math.min(maxThreads / threadsPerSM * 100, 100);
}

export default function RegisterPressureAnalyzer() {
  const [tileSize, setTileSize] = useState(16);
  const [dtype, setDtype] = useState<'fp16' | 'fp32'>('fp16');
  const [smemUsage, setSmemUsage] = useState(0);

  const regs = useMemo(() => calcRegisters(tileSize, dtype), [tileSize, dtype]);
  const occupancy = useMemo(() => calcOccupancy(regs), [regs]);

  const barHeight = Math.min(occupancy, 100);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">寄存器压力分析器</h2>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Tile 大小</label>
          <div className="flex gap-2">
            {[8, 16, 32].map(s => (
              <button key={s} onClick={() => setTileSize(s)}
                className={`px-3 py-1 rounded text-sm ${tileSize === s ? 'bg-blue-600' : 'bg-gray-700'}`}>
                {s}×{s}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">数据类型</label>
          <div className="flex gap-2">
            {(['fp16', 'fp32'] as const).map(d => (
              <button key={d} onClick={() => setDtype(d)}
                className={`px-3 py-1 rounded text-sm uppercase ${dtype === d ? 'bg-blue-600' : 'bg-gray-700'}`}>
                {d}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-6">
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-400">{regs}</div>
          <div className="text-xs text-gray-400">寄存器/线程</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-400">{occupancy.toFixed(1)}%</div>
          <div className="text-xs text-gray-400">最大占用率</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-400">{Math.floor(65536 / regs)}</div>
          <div className="text-xs text-gray-400">最大线程数/SM</div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 mb-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">占用率</span>
          <span className="text-white">{occupancy.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-4">
          <div className="h-4 rounded-full transition-all duration-300"
            style={{
              width: `${barHeight}%`,
              background: barHeight > 75 ? '#10B981' : barHeight > 50 ? '#F59E0B' : '#EF4444'
            }} />
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 text-xs">
        {['Fragment A', 'Fragment B', 'Accumulator', 'Overhead'].map((label, i) => {
          const values = [tileSize * tileSize, tileSize * tileSize, tileSize * tileSize, 32];
          const colors = ['#3B82F6', '#8B5CF6', '#F59E0B', '#6B7280'];
          return (
            <div key={i} className="bg-gray-800 rounded p-2 text-center">
              <div className="font-mono" style={{ color: colors[i] }}>{values[i]}</div>
              <div className="text-gray-400 mt-0.5">{label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

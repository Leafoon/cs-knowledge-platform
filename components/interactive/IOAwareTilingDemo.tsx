'use client';

import { useState } from 'react';

export default function IOAwareTilingDemo() {
  const [showSram, setShowSram] = useState(true);

  const sramTiles = Array.from({ length: 6 }, (_, i) => ({
    id: i, x: (i % 3) * 40 + 10, y: Math.floor(i / 3) * 30 + 10,
  }));

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">IO 感知分块：SRAM vs HBM</h2>
        <button onClick={() => setShowSram(!showSram)}
          className="px-3 py-1 bg-blue-600 rounded text-sm hover:bg-blue-500">
          {showSram ? '查看 HBM 访问' : '查看 SRAM 分块'}
        </button>
      </div>

      <div className="flex gap-4 mb-6">
        <div className="flex-1">
          <div className="text-sm text-gray-400 mb-2">
            {showSram ? 'SRAM 分块（共享内存）' : 'HBM 访问模式（全局内存）'}
          </div>
          <svg viewBox="0 0 300 200" className="w-full bg-black rounded-lg">
            {showSram ? (
              <>
                <text x="10" y="15" fill="#6B7280" fontSize="10">SRAM 192KB</text>
                <rect x="10" y="22" width="280" height="170" rx="4" fill="none" stroke="#374151" strokeWidth="1" strokeDasharray="4,2" />
                {sramTiles.map(t => (
                  <g key={t.id}>
                    <rect x={t.x + 20} y={t.y + 15} width="35" height="25" rx="3"
                      fill="#3B82F620" stroke="#3B82F6" strokeWidth="1" />
                    <text x={t.x + 37} y={t.y + 31} fill="#60A5FA" fontSize="8" textAnchor="middle">
                      Q[{t.id}]K[{t.id}]
                    </text>
                  </g>
                ))}
                <text x="150" y="198" fill="#3B82F6" fontSize="9" textAnchor="middle">每轮加载 6 个 tile 到 SRAM（12KB）</text>
              </>
            ) : (
              <>
                <text x="10" y="15" fill="#6B7280" fontSize="10">HBM 80GB</text>
                <rect x="10" y="22" width="280" height="170" rx="4" fill="none" stroke="#374151" strokeWidth="1" strokeDasharray="4,2" />
                {Array.from({ length: 24 }, (_, i) => {
                  const x = (i % 8) * 34 + 15;
                  const y = Math.floor(i / 8) * 50 + 30;
                  const accessed = Math.floor(i / 4) <= Math.floor(sramTiles.length / 3);
                  return (
                    <g key={i}>
                      <rect x={x} y={y} width="30" height="20" rx="2"
                        fill={accessed ? '#EF444420' : '#37415120'}
                        stroke={accessed ? '#EF4444' : '#374151'} strokeWidth={accessed ? 1.5 : 0.5} />
                      <text x={x + 15} y={y + 13} fill={accessed ? '#EF4444' : '#6B7280'} fontSize="6" textAnchor="middle">
                        {i}
                      </text>
                    </g>
                  );
                })}
                <text x="150" y="198" fill="#EF4444" fontSize="9" textAnchor="middle">传统方式：每次访问整个 Q/K 矩阵</text>
              </>
            )}
          </svg>
        </div>

        <div className="w-56 space-y-3">
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <div className="font-bold text-blue-400 mb-1">标准注意力 IO</div>
            <div className="text-gray-400">HBM 访问: <span className="text-red-400">O(N²)</span></div>
            <div className="text-gray-400">计算: O(N²)</div>
            <div className="text-gray-400">IO 比: <span className="text-yellow-400">受限于内存带宽</span></div>
          </div>
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <div className="font-bold text-green-400 mb-1">FlashAttention IO</div>
            <div className="text-gray-400">HBM 访问: <span className="text-green-400">O(N²d²/M)</span></div>
            <div className="text-gray-400">计算: O(N²d)</div>
            <div className="text-gray-400">IO 比: <span className="text-green-400">计算密集型</span></div>
          </div>
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <div className="text-gray-400">
              SRAM: 192KB（A100）<br/>
              HBM: 80GB<br/>
              <span className="text-blue-400">带宽比: 19:1</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

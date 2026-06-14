'use client';

import { useState } from 'react';

const ranks = ['Rank 0', 'Rank 1', 'Rank 2', 'Rank 3'];
const expertsPerRank = 2;

export default function MoESchedulingOptimization() {
  const [phase, setPhase] = useState<'dispatch' | 'compute' | 'combine'>('dispatch');
  const [activeRoute, setActiveRoute] = useState<number | null>(null);

  const phases = [
    { key: 'dispatch', label: '分发阶段', desc: 'Token 通过 All-to-All 分发到各 Rank 的专家' },
    { key: 'compute', label: '计算阶段', desc: '每个 Rank 上的专家独立处理分配的 Token' },
    { key: 'combine', label: '合并阶段', desc: 'All-to-All 收集结果，按原始顺序重组' },
  ];

  const dispatchRoutes = [
    { fromRank: 0, toRank: 1, tokens: 128 },
    { fromRank: 0, toRank: 2, tokens: 96 },
    { fromRank: 1, toRank: 0, tokens: 112 },
    { fromRank: 1, toRank: 3, tokens: 64 },
    { fromRank: 2, toRank: 0, tokens: 88 },
    { fromRank: 2, toRank: 3, tokens: 144 },
    { fromRank: 3, toRank: 1, tokens: 104 },
    { fromRank: 3, toRank: 2, tokens: 80 },
  ];

  const getRouteColor = (idx: number) => {
    if (activeRoute === null) return '#4b5563';
    if (activeRoute === idx) return '#facc15';
    return '#374151';
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">MoE 调度优化 - All-to-All 通信</h2>
      <p className="text-gray-400 text-sm mb-4">展示混合专家模型中跨 Rank 的 All-to-All 通信模式</p>

      <div className="flex gap-2 mb-6">
        {phases.map((p) => (
          <button
            key={p.key}
            onClick={() => setPhase(p.key as typeof phase)}
            className={`px-4 py-2 rounded text-sm font-medium transition-all ${
              phase === p.key ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="text-sm text-gray-400 mb-4">{phases.find(p => p.key === phase)?.desc}</div>

      <div className="relative h-72 mb-4">
        <div className="flex justify-between px-8">
          {ranks.map((rank, ri) => (
            <div key={ri} className="flex flex-col items-center">
              <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 w-24">
                <div className="text-xs text-gray-400 text-center mb-1">{rank}</div>
                <div className="text-[10px] text-gray-500 text-center">
                  {expertsPerRank} 专家
                </div>
                <div className="mt-2 space-y-1">
                  {Array.from({ length: expertsPerRank }).map((_, ei) => (
                    <div key={ei} className="bg-blue-900/50 rounded px-2 py-0.5 text-[10px] text-center">
                      E{ri * expertsPerRank + ei}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 800 288">
          {phase === 'dispatch' && dispatchRoutes.map((route, idx) => {
            const startX = 80 + route.fromRank * ((800 - 160) / 3);
            const endX = 80 + route.toRank * ((800 - 160) / 3);
            const startY = 60 + route.fromRank * 10;
            const endY = 60 + route.toRank * 10;

            return (
              <g key={idx}>
                <line
                  x1={startX}
                  y1={startY}
                  x2={endX}
                  y2={endY + 160}
                  stroke={getRouteColor(idx)}
                  strokeWidth={activeRoute === idx ? 2.5 : 1}
                  strokeDasharray={activeRoute === idx ? 'none' : '3,3'}
                  className="transition-all cursor-pointer"
                  style={{ pointerEvents: 'stroke' }}
                  onMouseEnter={() => setActiveRoute(idx)}
                  onMouseLeave={() => setActiveRoute(null)}
                />
                {activeRoute === idx && (
                  <text
                    x={(startX + endX) / 2}
                    y={(startY + endY + 160) / 2 - 5}
                    fill="#facc15"
                    fontSize="10"
                    textAnchor="middle"
                  >
                    {route.tokens} tokens
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">通信量</div>
          <div className="text-lg font-bold text-blue-400">816 tokens</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">通信延迟</div>
          <div className="text-lg font-bold text-yellow-400">2.3ms</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">带宽利用率</div>
          <div className="text-lg font-bold text-green-400">87.5%</div>
        </div>
      </div>
    </div>
  );
}

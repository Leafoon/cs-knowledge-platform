'use client';

import { useState } from 'react';

const experts = [
  { id: 0, name: '专家 0', color: 'bg-blue-500', load: 85 },
  { id: 1, name: '专家 1', color: 'bg-green-500', load: 72 },
  { id: 2, name: '专家 2', color: 'bg-purple-500', load: 91 },
  { id: 3, name: '专家 3', color: 'bg-orange-500', load: 45 },
  { id: 4, name: '专家 4', color: 'bg-red-500', load: 63 },
  { id: 5, name: '专家 5', color: 'bg-teal-500', load: 78 },
];

const tokenRoutes = [
  { from: 0, to: 2, weight: 0.4 },
  { from: 0, to: 5, weight: 0.35 },
  { from: 1, to: 0, weight: 0.5 },
  { from: 1, to: 3, weight: 0.25 },
  { from: 2, to: 1, weight: 0.3 },
  { from: 2, to: 4, weight: 0.45 },
  { from: 3, to: 2, weight: 0.55 },
  { from: 3, to: 5, weight: 0.2 },
];

export default function TokenDispatchFlow() {
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [balanceMetric, setBalanceMetric] = useState(0.82);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">Token 调度流程 - MoE 负载均衡</h2>
      <p className="text-gray-400 text-sm mb-4">展示 Token 如何被分发到不同的专家网络，以及负载均衡机制</p>

      <div className="flex items-center gap-4 mb-6">
        <label className="text-sm text-gray-300">负载均衡系数:</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={balanceMetric}
          onChange={(e) => setBalanceMetric(parseFloat(e.target.value))}
          className="w-48"
        />
        <span className={`text-sm font-mono ${balanceMetric > 0.7 ? 'text-green-400' : balanceMetric > 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
          {balanceMetric.toFixed(2)}
        </span>
      </div>

      <div className="relative h-64 mb-4">
        <div className="absolute left-0 top-1/2 -translate-y-1/2 space-y-2">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              onClick={() => setSelectedToken(i)}
              className={`w-12 h-8 rounded flex items-center justify-center text-xs font-bold cursor-pointer transition-all ${
                selectedToken === i ? 'bg-yellow-400 text-black scale-110' : 'bg-yellow-600 text-white'
              }`}
            >
              T{i}
            </div>
          ))}
        </div>

        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 256">
          {tokenRoutes.map((route, idx) => {
            const isActive = selectedToken !== null && route.from === selectedToken;
            const opacity = selectedToken === null ? 0.3 : isActive ? 0.9 : 0.1;
            const x1 = 60;
            const y1 = 32 + route.from * 64;
            const x2 = 420;
            const y2 = 16 + (route.to * 40);
            const cx1 = 200;
            const cy1 = y1;
            const cx2 = 320;
            const cy2 = y2;

            return (
              <path
                key={idx}
                d={`M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`}
                fill="none"
                stroke={isActive ? '#facc15' : '#6b7280'}
                strokeWidth={isActive ? 3 : 1}
                strokeDasharray={isActive ? 'none' : '4,4'}
                opacity={opacity}
              />
            );
          })}
        </svg>

        <div className="absolute right-0 top-1/2 -translate-y-1/2 space-y-1">
          {experts.map((expert) => (
            <div key={expert.id} className="flex items-center gap-2">
              <div className={`w-28 h-8 ${expert.color} rounded flex items-center justify-center text-xs font-bold`}>
                {expert.name}
              </div>
              <div className="w-20 h-2 bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full ${expert.color} transition-all`}
                  style={{ width: `${expert.load}%` }}
                />
              </div>
              <span className="text-xs text-gray-400 w-8">{expert.load}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">总 Token 数</div>
          <div className="text-lg font-bold text-white">1024</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">Top-K 选择</div>
          <div className="text-lg font-bold text-blue-400">K = 2</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">负载方差</div>
          <div className={`text-lg font-bold ${balanceMetric > 0.7 ? 'text-green-400' : 'text-red-400'}`}>
            {(1 - balanceMetric).toFixed(3)}
          </div>
        </div>
      </div>
    </div>
  );
}

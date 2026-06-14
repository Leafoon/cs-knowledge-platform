'use client';

import { useState } from 'react';

const patterns = [
  { name: '顺序读取', achieved: 856, theoretical: 900, color: 'bg-green-500' },
  { name: '随机读取', achieved: 127, theoretical: 900, color: 'bg-red-500' },
  { name: '合并写入', achieved: 743, theoretical: 900, color: 'bg-blue-500' },
  { name: '非合并写入', achieved: 89, theoretical: 900, color: 'bg-orange-500' },
  { name: '共享内存', achieved: 12000, theoretical: 19000, color: 'bg-purple-500' },
];

export function BandwidthUtilizationChart() {
  const [selectedPattern, setSelectedPattern] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">带宽利用率对比</h2>
      
      <div className="space-y-4">
        {patterns.map((p, i) => {
          const utilization = (p.achieved / p.theoretical) * 100;
          return (
            <div
              key={p.name}
              onClick={() => setSelectedPattern(selectedPattern === i ? null : i)}
              className={`p-3 rounded-lg cursor-pointer transition-all ${
                selectedPattern === i ? 'bg-gray-50 ring-2 ring-blue-300' : 'hover:bg-gray-50'
              }`}
            >
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">{p.name}</span>
                <span className="font-mono text-gray-800">{p.achieved} / {p.theoretical} GB/s</span>
              </div>
              <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full ${p.color} transition-all`}
                  style={{ width: `${Math.min(utilization, 100)}%` }}
                />
              </div>
              {selectedPattern === i && (
                <div className="mt-2 text-sm text-gray-600">
                  利用率: {utilization.toFixed(1)}%
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
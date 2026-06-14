'use client';

import { useState } from 'react';

export function ExpertGEMMScheduling() {
  const [time, setTime] = useState(0);

  const experts = [
    { id: 0, size: '4096x4096', start: 0, duration: 3 },
    { id: 1, size: '2048x4096', start: 0, duration: 2 },
    { id: 2, size: '1024x4096', start: 2, duration: 1 },
  ];

  const maxTime = 5;

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">专家 GEMM 调度</h2>
      
      <div className="flex gap-2 mb-4">
        {Array.from({ length: maxTime }, (_, i) => (
          <button
            key={i}
            onClick={() => setTime(i)}
            className={`w-8 h-8 rounded text-sm ${
              time === i ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            T{i}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {experts.map((e) => (
          <div key={e.id} className="flex items-center gap-2">
            <div className="w-20 text-sm text-gray-600">专家{e.id}</div>
            <div className="flex-1 h-8 bg-gray-100 rounded relative overflow-hidden">
              <div
                className={`absolute h-full rounded transition-all ${
                  time >= e.start && time < e.start + e.duration
                    ? 'bg-green-500'
                    : 'bg-gray-300'
                }`}
                style={{
                  left: `${(e.start / maxTime) * 100}%`,
                  width: `${(e.duration / maxTime) * 100}%`,
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-700">
                {e.size}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm">
        <div className="text-gray-600">
          当前时间步 T{time}: {
            experts
              .filter((e) => time >= e.start && time < e.start + e.duration)
              .map((e) => `专家${e.id}`)
              .join(', ') || '无活跃专家'
          }
        </div>
      </div>

      <div className="mt-4 p-3 bg-purple-50 rounded-lg text-sm text-purple-700">
        💡 动态批处理可以根据专家负载调整调度策略
      </div>
    </div>
  );
}
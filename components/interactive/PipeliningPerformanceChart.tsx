'use client';

import { useState } from 'react';

const data = [
  { stages: 1, tflops: 12.4, color: '#EF4444', label: '无流水线' },
  { stages: 2, tflops: 21.8, color: '#F59E0B', label: '2级流水线' },
  { stages: 3, tflops: 28.6, color: '#3B82F6', label: '3级流水线' },
  { stages: 4, tflops: 31.2, color: '#10B981', label: '4级流水线' },
];

const maxTflops = 35;

export default function PipeliningPerformanceChart() {
  const [hoveredBar, setHoveredBar] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">流水线级数 vs GEMM 吞吐量</h2>
      <p className="text-sm text-gray-400 mb-6">A100-80GB · FP16 · M=N=K=4096 · Tile 128×128</p>

      <div className="relative h-64 mb-4">
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-500 pr-2"
          style={{ width: '40px' }}>
          {[35, 28, 21, 14, 7, 0].map(v => <span key={v}>{v}</span>)}
        </div>

        <div className="ml-12 h-full flex items-end gap-6 justify-center">
          {data.map((d, i) => {
            const height = (d.tflops / maxTflops) * 100;
            const isHovered = hoveredBar === i;
            return (
              <div key={i} className="flex flex-col items-center" style={{ width: '80px' }}
                onMouseEnter={() => setHoveredBar(i)} onMouseLeave={() => setHoveredBar(null)}>
                <div className="text-sm font-bold mb-1 transition-all" style={{ color: d.color, opacity: isHovered ? 1 : 0.7 }}>
                  {d.tflops} TFLOPS
                </div>
                <div className="w-full rounded-t-lg transition-all duration-300"
                  style={{
                    height: `${height}%`,
                    backgroundColor: d.color,
                    opacity: isHovered ? 1 : 0.7,
                    transform: isHovered ? 'scaleY(1.03)' : 'scaleY(1)',
                  }} />
                <div className="mt-2 text-center">
                  <div className="text-sm font-bold">{d.stages}级</div>
                  <div className="text-[10px] text-gray-400">{d.label}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 text-xs">
        {data.map((d, i) => (
          <div key={i} className="bg-gray-800 rounded p-2 text-center">
            <div className="text-gray-400">加速比</div>
            <div className="font-bold" style={{ color: d.color }}>
              {(d.tflops / data[0].tflops).toFixed(1)}×
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

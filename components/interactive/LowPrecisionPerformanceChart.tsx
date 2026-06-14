'use client';

import { useState } from 'react';

const data = [
  { format: 'FP32', tflops: 19.5, color: '#EF4444', width: 4 },
  { format: 'TF32', tflops: 28.3, color: '#F97316', width: 3 },
  { format: 'FP16', tflops: 31.2, color: '#F59E0B', width: 2 },
  { format: 'BF16', tflops: 31.0, color: '#EAB308', width: 2 },
  { format: 'INT8', tflops: 62.4, color: '#3B82F6', width: 1 },
  { format: 'INT4', tflops: 124.8, color: '#8B5CF6', width: 0.5 },
];

const maxVal = 140;

export default function LowPrecisionPerformanceChart() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">低精度 GEMM 吞吐量对比</h2>
      <p className="text-sm text-gray-400 mb-6">NVIDIA H100 · 理论峰值 TFLOPS</p>

      <div className="space-y-3">
        {data.map((d, i) => {
          const pct = (d.tflops / maxVal) * 100;
          const isSelected = selected === i;
          return (
            <div key={i} className="cursor-pointer" onClick={() => setSelected(selected === i ? null : i)}>
              <div className="flex items-center gap-3">
                <span className="w-12 text-sm font-mono text-gray-400">{d.format}</span>
                <div className="flex-1 bg-gray-800 rounded-full h-6 relative overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: d.color,
                      opacity: isSelected ? 1 : 0.8,
                      boxShadow: isSelected ? `0 0 12px ${d.color}60` : 'none',
                    }}>
                    <span className="text-xs font-bold text-white drop-shadow">{d.tflops} TFLOPS</span>
                  </div>
                </div>
                <span className="w-16 text-xs text-gray-500">{d.width}B/param</span>
              </div>
              {isSelected && (
                <div className="ml-15 mt-1 p-2 bg-gray-800 rounded text-xs text-gray-300 ml-15">
                  相对 FP32 加速: <span className="text-green-400 font-bold">{(d.tflops / data[0].tflops).toFixed(1)}×</span>
                  · 内存节省: <span className="text-blue-400 font-bold">{(4 / d.width).toFixed(0)}×</span>
                  · 适用: {i < 3 ? '训练/推理' : '推理优化'}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">FP32 基线</div>
          <div className="font-bold text-red-400">19.5 TFLOPS</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">FP16 Tensor Core</div>
          <div className="font-bold text-yellow-400">31.2 TFLOPS</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">INT4 最高</div>
          <div className="font-bold text-purple-400">124.8 TFLOPS</div>
        </div>
      </div>
    </div>
  );
}

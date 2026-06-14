'use client';

import { useState } from 'react';

const benchmarks = [
  {
    kernel: '3×3',
    direct: { gflops: 1250, ms: 4.8 },
    im2col: { gflops: 3200, ms: 1.9 },
    winograd: { gflops: 4100, ms: 1.5 },
  },
  {
    kernel: '5×5',
    direct: { gflops: 980, ms: 6.1 },
    im2col: { gflops: 2800, ms: 2.1 },
    winograd: { gflops: 3500, ms: 1.7 },
  },
  {
    kernel: '1×1',
    direct: { gflops: 4200, ms: 1.4 },
    im2col: { gflops: 4200, ms: 1.4 },
    winograd: { gflops: 0, ms: 0 },
  },
  {
    kernel: '7×7',
    direct: { gflops: 750, ms: 7.9 },
    im2col: { gflops: 2400, ms: 2.5 },
    winograd: { gflops: 2900, ms: 2.0 },
  },
];

export function ConvPerformanceBenchmark() {
  const [selectedKernel, setSelectedKernel] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">卷积性能基准测试</h2>
      
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gradient-to-r from-blue-500 to-purple-500 text-white">
              <th className="p-3 text-left">卷积核</th>
              <th className="p-3 text-left">直接卷积</th>
              <th className="p-3 text-left">Im2Col</th>
              <th className="p-3 text-left">Winograd</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((b, i) => (
              <tr
                key={b.kernel}
                onClick={() => setSelectedKernel(selectedKernel === i ? null : i)}
                className={`border-b cursor-pointer transition-all ${
                  selectedKernel === i ? 'bg-blue-50' : 'hover:bg-gray-50'
                }`}
              >
                <td className="p-3 font-semibold">{b.kernel}</td>
                <td className="p-3">
                  <div className="text-sm">{b.direct.gflops} GFLOPS</div>
                  <div className="text-xs text-gray-500">{b.direct.ms}ms</div>
                </td>
                <td className="p-3">
                  <div className="text-sm">{b.im2col.gflops} GFLOPS</div>
                  <div className="text-xs text-gray-500">{b.im2col.ms}ms</div>
                </td>
                <td className="p-3">
                  <div className="text-sm">
                    {b.winograd.gflops > 0 ? `${b.winograd.gflops} GFLOPS` : '不适用'}
                  </div>
                  <div className="text-xs text-gray-500">
                    {b.winograd.ms > 0 ? `${b.winograd.ms}ms` : '-'}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 1×1卷积直接等于GEMM，Winograd仅适用于3×3卷积
      </div>
    </div>
  );
}
'use client';

import { useState } from 'react';

const rooflineData = [
  { flops: 1e9, mem: 1e12, label: '1 GFLOPS' },
  { flops: 1e10, mem: 1e11, label: '10 GFLOPS' },
  { flops: 1e11, mem: 1e10, label: '100 GFLOPS' },
  { flops: 1e12, mem: 1e9, label: '1 TFLOPS' },
];

const kernels = [
  { name: 'GEMM (大矩阵)', intensity: 128, achieved: 320, peak: 400, color: 'bg-blue-500' },
  { name: 'GEMM (小矩阵)', intensity: 32, achieved: 85, peak: 400, color: 'bg-blue-400' },
  { name: 'Flash Attention', intensity: 64, achieved: 210, peak: 400, color: 'bg-green-500' },
  { name: 'LayerNorm', intensity: 4, achieved: 18, peak: 40, color: 'bg-yellow-500' },
  { name: 'Softmax', intensity: 2, achieved: 8, peak: 40, color: 'bg-orange-500' },
  { name: 'Element-wise', intensity: 1, achieved: 3.5, peak: 4, color: 'bg-red-500' },
];

export default function PerformanceCeilingChart() {
  const [selectedKernel, setSelectedKernel] = useState<string | null>(null);
  const [showRoof, setShowRoof] = useState(true);

  const peakCompute = 400;
  const peakBandwidth = 3.35;
  const ridgePoint = peakCompute / peakBandwidth;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">性能天花板 - Roofline 模型</h2>
      <p className="text-gray-400 text-sm mb-4">TileLang 内核在 Roofline 模型中的性能表现</p>

      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-xs text-gray-400">
          <input
            type="checkbox"
            checked={showRoof}
            onChange={(e) => setShowRoof(e.target.checked)}
            className="rounded"
          />
          显示 Roofline 上界
        </label>
      </div>

      <div className="relative bg-gray-800 rounded-lg p-4 h-80">
        <div className="absolute inset-0 p-8">
          <svg viewBox="0 0 600 280" className="w-full h-full">
            {/* Grid lines */}
            {[0.1, 1, 10, 100].map((v, i) => (
              <g key={i}>
                <line
                  x1={80 + (Math.log10(v) + 1) * 40}
                  y1={20}
                  x2={80 + (Math.log10(v) + 1) * 40}
                  y2={240}
                  stroke="#374151"
                  strokeWidth="0.5"
                />
                <text
                  x={80 + (Math.log10(v) + 1) * 40}
                  y={260}
                  fill="#6b7280"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {v}
                </text>
              </g>
            ))}

            {/* Y axis labels */}
            {[1, 10, 100, 400].map((v, i) => (
              <g key={i}>
                <line
                  x1={80}
                  y1={240 - (v / 400) * 220}
                  x2={560}
                  y2={240 - (v / 400) * 220}
                  stroke="#374151"
                  strokeWidth="0.5"
                />
                <text
                  x={70}
                  y={244 - (v / 400) * 220}
                  fill="#6b7280"
                  fontSize="10"
                  textAnchor="end"
                >
                  {v}
                </text>
              </g>
            ))}

            {/* Roofline */}
            {showRoof && (
              <path
                d={`M 80 22 L 240 22 L 560 ${240 - (peakCompute / ridgePoint) * 220 / 400}`}
                fill="none"
                stroke="#facc15"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
            )}

            {/* Kernel points */}
            {kernels.map((kernel, idx) => {
              const x = 80 + (Math.log10(kernel.intensity) + 1) * 40;
              const y = 240 - (kernel.achieved / 400) * 220;
              const isSelected = selectedKernel === kernel.name;

              return (
                <g
                  key={idx}
                  className="cursor-pointer"
                  onClick={() => setSelectedKernel(isSelected ? null : kernel.name)}
                >
                  <circle
                    cx={x}
                    cy={y}
                    r={isSelected ? 10 : 7}
                    fill={isSelected ? '#facc15' : kernel.color.replace('bg-', '#').replace('500', '').replace('400', '')}
                    stroke={isSelected ? '#fff' : 'none'}
                    strokeWidth="2"
                  />
                  {isSelected && (
                    <g>
                      <rect
                        x={x + 15}
                        y={y - 40}
                        width="120"
                        height="50"
                        fill="#1f2937"
                        stroke="#374151"
                        rx="4"
                      />
                      <text x={x + 22} y={y - 24} fill="#fff" fontSize="10">
                        {kernel.name}
                      </text>
                      <text x={x + 22} y={y - 10} fill="#9ca3af" fontSize="9">
                        算术强度: {kernel.intensity}
                      </text>
                      <text x={x + 22} y={y + 2} fill="#34d399" fontSize="9">
                        达到: {kernel.achieved} GFLOPS
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* Ridge point marker */}
            {showRoof && (
              <>
                <circle cx={240} cy={22} r={4} fill="#facc15" />
                <text x={245} y={18} fill="#facc15" fontSize="9">
                  Ridge Point
                </text>
              </>
            )}

            {/* Axis labels */}
            <text x={300} y={275} fill="#9ca3af" fontSize="11" textAnchor="middle">
              算术强度 (FLOPS/Byte)
            </text>
            <text
              x={15}
              y={140}
              fill="#9ca3af"
              fontSize="11"
              textAnchor="middle"
              transform="rotate(-90, 15, 140)"
            >
              性能 (GFLOPS)
            </text>
          </svg>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-xs">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">计算上界</div>
          <div className="text-lg font-bold text-yellow-400">{peakCompute} GFLOPS</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">内存带宽上界</div>
          <div className="text-lg font-bold text-green-400">{peakBandwidth} TB/s</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">Ridge Point</div>
          <div className="text-lg font-bold text-blue-400">{ridgePoint.toFixed(1)}</div>
        </div>
      </div>
    </div>
  );
}

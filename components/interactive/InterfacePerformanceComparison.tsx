'use client';

import React, { useState } from 'react';

interface BenchmarkResult {
  level: string;
  tflops: number;
  color: string;
  utilization: number;
}

export function InterfacePerformanceComparison() {
  const [showDetails, setShowDetails] = useState(false);

  const benchmarks: BenchmarkResult[] = [
    { level: 'Beginner', tflops: 45.2, color: '#10B981', utilization: 72 },
    { level: 'Developer', tflops: 78.5, color: '#F59E0B', utilization: 91 },
    { level: 'Expert', tflops: 82.3, color: '#EF4444', utilization: 95 },
  ];

  const maxTflops = 100;

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">接口性能对比 (GEMM)</h2>
      
      <div className="space-y-6">
        {benchmarks.map((bench, i) => (
          <div key={i} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-white font-bold">{bench.level}</span>
              <span className="text-gray-400">{bench.tflops} TFLOPS</span>
            </div>
            
            <div className="relative h-8 bg-gray-800 rounded-lg overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 rounded-lg transition-all duration-1000"
                style={{
                  width: `${(bench.tflops / maxTflops) * 100}%`,
                  backgroundColor: bench.color,
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-white font-bold text-sm">
                {bench.utilization}% 利用率
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-8 grid grid-cols-3 gap-4">
        {benchmarks.map((bench, i) => (
          <div
            key={i}
            className="p-4 rounded-lg border"
            style={{ borderColor: bench.color }}
          >
            <div className="text-center">
              <div className="text-3xl font-bold" style={{ color: bench.color }}>
                {bench.tflops}
              </div>
              <div className="text-gray-400 text-sm">TFLOPS</div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-white font-bold">性能提升分析</span>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-blue-400 text-sm hover:underline"
          >
            {showDetails ? '收起' : '展开详情'}
          </button>
        </div>
        
        {showDetails && (
          <div className="text-gray-300 text-sm space-y-2">
            <p>• Beginner → Developer: +73.7% 性能提升</p>
            <p>• Developer → Expert: +4.8% 性能提升</p>
            <p>• Expert 比 Beginner 快 1.82x</p>
            <p>• Expert 已接近硬件理论峰值 (95% 利用率)</p>
          </div>
        )}
      </div>
    </div>
  );
}

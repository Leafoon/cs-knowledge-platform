'use client';

import { useState } from 'react';

const steps = [
  { label: '加载 INT4 权重', color: '#EF4444', icon: '📦', desc: '从全局内存加载 4-bit 量化权重到寄存器' },
  { label: '反量化', color: '#F59E0B', icon: '🔧', desc: 'W_fp16 = (W_int4 - zero_point) × scale' },
  { label: 'FP16 矩阵乘', color: '#3B82F6', icon: '✖️', desc: '使用 Tensor Core 执行 FP16 MMA 指令' },
  { label: 'FP32 累加', color: '#10B981', icon: '➕', desc: '累加器保持 FP32 精度避免溢出' },
];

export default function DequantizeGEMMFlow() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">反量化 GEMM 流程</h2>

      <div className="flex flex-col gap-3 mb-6">
        {steps.map((s, i) => (
          <div key={i} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all cursor-pointer ${
                  i <= activeStep ? 'scale-110' : 'opacity-40'
                }`}
                style={{ backgroundColor: `${s.color}30`, border: `2px solid ${i <= activeStep ? s.color : '#374151'}` }}
                onClick={() => setActiveStep(i)}>
                {s.icon}
              </div>
              {i < steps.length - 1 && (
                <div className={`w-0.5 h-8 my-1 ${i < activeStep ? 'bg-blue-500' : 'bg-gray-700'}`} />
              )}
            </div>
            <div className={`flex-1 p-3 rounded-lg border transition-all ${
              i === activeStep ? 'border-opacity-100 bg-opacity-10' : 'border-gray-700 opacity-50'
            }`} style={{ borderColor: i <= activeStep ? s.color : undefined, backgroundColor: `${i === activeStep ? s.color : '#000'}10` }}>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-bold text-sm" style={{ color: s.color }}>{s.label}</span>
                <span className="text-xs text-gray-500">Step {i + 1}</span>
              </div>
              <p className="text-xs text-gray-400">{s.desc}</p>
              {i === activeStep && (
                <div className="mt-2 bg-black/40 rounded p-2 font-mono text-xs">
                  {i === 0 && <span className="text-red-400">ld.global.v4.u8 {'{'}w0, w1, w2, w3{'}'}, [addr];</span>}
                  {i === 1 && <span className="text-yellow-400">w_fp16 = (w_int4 - zp) * scale_per_channel;</span>}
                  {i === 2 && <span className="text-blue-400">mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ...</span>}
                  {i === 3 && <span className="text-green-400">acc_fp32 += (float)mma_result;</span>}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300">
        <b className="text-white">关键优化：</b>反量化与计算重叠，使用向量化 load 一次加载 16 个 INT4 权重（8 字节），per-channel scale 保留在常量内存中。
      </div>
    </div>
  );
}

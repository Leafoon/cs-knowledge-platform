'use client';

import { useState } from 'react';

export function WinogradTransformFlow() {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: '输入变换 (BᵀdB)',
      description: '将输入tile从4x4变换到4x4域',
      matrix: 'Bᵀ = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]',
    },
    {
      title: '滤波器变换 (GgGᵀ)',
      description: '将卷积核从3x3变换到4x4域',
      matrix: 'G = [1 0 0; 0.5 0.5 0.5; 0.5 -0.5 0.5; 0 0 1]',
    },
    {
      title: '逐点乘法',
      description: '在变换域进行16次逐点乘法',
      matrix: 'U = GgGᵀ ⊙ BᵀdB',
    },
    {
      title: '输出变换 (AᵀmA)',
      description: '将结果从4x4域变换回2x2输出',
      matrix: 'Aᵀ = [1 1 1 0; 0 1 -1 -1]',
    },
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Winograd F(2x2,3x3) 变换流程</h2>
      
      <div className="flex gap-2 mb-6">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`flex-1 py-2 rounded-lg transition-colors text-sm ${
              step === i ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div className="p-4 bg-gray-50 rounded-xl mb-4">
        <div className="text-sm text-gray-500 mb-1">{steps[step].description}</div>
        <div className="font-mono text-sm text-gray-800 bg-white p-2 rounded">{steps[step].matrix}</div>
      </div>

      <div className="flex justify-center gap-4 items-center">
        <div className="w-24 h-24 bg-blue-100 rounded-lg flex items-center justify-center text-center">
          <div>
            <div className="text-xs text-blue-500">输入</div>
            <div className="font-bold">4×4</div>
          </div>
        </div>
        <div className="text-2xl">→</div>
        <div className="w-24 h-24 bg-purple-100 rounded-lg flex items-center justify-center text-center">
          <div>
            <div className="text-xs text-purple-500">变换</div>
            <div className="font-bold">BᵀdB</div>
          </div>
        </div>
        <div className="text-2xl">→</div>
        <div className="w-24 h-24 bg-green-100 rounded-lg flex items-center justify-center text-center">
          <div>
            <div className="text-xs text-green-500">逐点乘</div>
            <div className="font-bold">16次</div>
          </div>
        </div>
        <div className="text-2xl">→</div>
        <div className="w-24 h-24 bg-orange-100 rounded-lg flex items-center justify-center text-center">
          <div>
            <div className="text-xs text-orange-500">输出</div>
            <div className="font-bold">2×2</div>
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-purple-50 rounded-lg text-sm text-purple-700">
        💡 Winograd将乘法次数从9次减少到4次，但增加了加法和变换开销
      </div>
    </div>
  );
}
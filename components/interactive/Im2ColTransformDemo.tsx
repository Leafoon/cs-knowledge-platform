'use client';

import { useState } from 'react';

export function Im2ColTransformDemo() {
  const [step, setStep] = useState(0);

  const input = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
  ];

  const kernel = [
    [1, 0],
    [0, 1],
  ];

  const columns = [
    [1, 2, 5, 6],
    [2, 3, 6, 7],
    [3, 4, 7, 8],
    [5, 6, 9, 10],
    [6, 7, 10, 11],
    [7, 8, 11, 12],
    [9, 10, 13, 14],
    [10, 11, 14, 15],
    [11, 12, 15, 16],
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Im2Col 变换演示</h2>
      
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setStep(0)}
          className={`px-3 py-1 rounded ${step === 0 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
        >
          输入矩阵
        </button>
        <button
          onClick={() => setStep(1)}
          className={`px-3 py-1 rounded ${step === 1 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
        >
          卷积核
        </button>
        <button
          onClick={() => setStep(2)}
          className={`px-3 py-1 rounded ${step === 2 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
        >
          列矩阵
        </button>
      </div>

      <div className="flex justify-center gap-8">
        {step === 0 && (
          <div>
            <div className="text-sm text-gray-500 mb-2 text-center">输入 (4×4)</div>
            <div className="grid grid-cols-4 gap-1">
              {input.flat().map((v, i) => (
                <div key={i} className="w-12 h-12 bg-blue-100 rounded flex items-center justify-center font-mono">
                  {v}
                </div>
              ))}
            </div>
          </div>
        )}

        {step === 1 && (
          <div>
            <div className="text-sm text-gray-500 mb-2 text-center">卷积核 (2×2)</div>
            <div className="grid grid-cols-2 gap-1">
              {kernel.flat().map((v, i) => (
                <div key={i} className="w-12 h-12 bg-purple-100 rounded flex items-center justify-center font-mono">
                  {v}
                </div>
              ))}
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <div className="text-sm text-gray-500 mb-2 text-center">列矩阵 (9×4)</div>
            <div className="grid grid-cols-4 gap-1">
              {columns.flat().map((v, i) => (
                <div key={i} className="w-12 h-12 bg-green-100 rounded flex items-center justify-center font-mono text-sm">
                  {v}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 Im2Col将卷积转换为矩阵乘法，可以复用高效的GEMM实现
      </div>
    </div>
  );
}
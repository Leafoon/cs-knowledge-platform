'use client';

import { useState } from 'react';

export function ReductionPatternDiagram() {
  const [step, setStep] = useState(0);

  const elements = ['1', '2', '3', '4', '5', '6', '7', '8'];
  const steps = [
    { title: '初始状态', data: elements },
    { title: '第1轮归约', data: ['1+2', '3+4', '5+6', '7+8'] },
    { title: '第2轮归约', data: ['(1+2)+(3+4)', '(5+6)+(7+8)'] },
    { title: '最终结果', data: ['1+2+3+4+5+6+7+8'] },
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">树形归约模式</h2>
      
      <div className="flex gap-2 mb-6">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              step === i ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div className="flex flex-col items-center gap-4">
        {steps.map((s, level) => (
          <div key={level} className={`flex gap-2 ${level <= step ? 'opacity-100' : 'opacity-30'}`}>
            {s.data.map((d, i) => (
              <div
                key={i}
                className={`px-3 py-2 rounded-lg text-sm font-mono ${
                  level === step
                    ? 'bg-blue-100 text-blue-700 ring-2 ring-blue-300'
                    : level < step
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-400'
                }`}
              >
                {d}
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="mt-6 p-3 bg-purple-50 rounded-lg text-sm text-purple-700">
        💡 树形归约将O(N)串行操作减少到O(log N)轮并行操作
      </div>
    </div>
  );
}
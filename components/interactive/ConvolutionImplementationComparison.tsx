'use client';

import { useState } from 'react';

const methods = [
  {
    name: '直接卷积',
    description: '直接在输入上滑动卷积核',
    flops: 'O(K²·C·H·W·M)',
    memory: 'O(H·W·C)',
    pros: ['直观易懂', '无需额外内存'],
    cons: ['难以优化', '循环嵌套多'],
    color: 'bg-red-500',
  },
  {
    name: 'Im2Col + GEMM',
    description: '将卷积转换为矩阵乘法',
    flops: 'O(K²·C·H·W·M)',
    memory: 'O(K²·C·H·W·M)',
    pros: ['复用GEMM优化', 'Tensor Core加速'],
    cons: ['内存开销大', '需要额外转换'],
    color: 'bg-blue-500',
  },
  {
    name: 'Winograd',
    description: '通过变换减少乘法次数',
    flops: 'O(2.25·C·H·W·M)',
    memory: 'O(H·W·C)',
    pros: ['乘法次数减少', '适合3x3卷积'],
    cons: ['数值稳定性', '仅适用特定卷积核'],
    color: 'bg-green-500',
  },
];

export function ConvolutionImplementationComparison() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">卷积实现方式对比</h2>
      
      <div className="grid grid-cols-3 gap-4">
        {methods.map((m, i) => (
          <div
            key={m.name}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
              selected === i ? 'border-blue-300 bg-blue-50' : 'border-transparent hover:bg-gray-50'
            }`}
          >
            <div className={`w-3 h-3 rounded-full ${m.color} mb-2`} />
            <div className="font-semibold text-gray-800 mb-1">{m.name}</div>
            <div className="text-sm text-gray-500 mb-3">{m.description}</div>
            
            <div className="text-xs space-y-1">
              <div><span className="text-gray-500">计算量:</span> {m.flops}</div>
              <div><span className="text-gray-500">内存:</span> {m.memory}</div>
            </div>

            {selected === i && (
              <div className="mt-3 pt-3 border-t border-gray-200 text-xs space-y-2">
                <div>
                  <div className="text-green-600 font-medium">优点:</div>
                  {m.pros.map((p) => (
                    <div key={p} className="text-gray-600">+ {p}</div>
                  ))}
                </div>
                <div>
                  <div className="text-red-600 font-medium">缺点:</div>
                  {m.cons.map((c) => (
                    <div key={c} className="text-gray-600">- {c}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
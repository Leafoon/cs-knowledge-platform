'use client';

import { useState } from 'react';

const fusionTypes = [
  {
    name: '逐点融合',
    description: '多个逐点操作合并为一个内核',
    example: 'ReLU + Scale + Bias',
    memoryReduction: 'N倍',
    complexity: '低',
    color: 'bg-green-500',
  },
  {
    name: '归约融合',
    description: '逐点操作与归约操作融合',
    example: 'LayerNorm (Mean + Variance + Normalize)',
    memoryReduction: '中等',
    complexity: '中',
    color: 'bg-blue-500',
  },
  {
    name: '生产者-消费者融合',
    description: '相邻算子通过共享内存通信',
    example: 'Conv + ReLU',
    memoryReduction: '高',
    complexity: '高',
    color: 'bg-purple-500',
  },
];

export function FusionStrategyComparison() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">融合策略对比</h2>
      
      <div className="space-y-3">
        {fusionTypes.map((type, i) => (
          <div
            key={type.name}
            onClick={() => setSelected(selected === i ? null : i)}
            className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
              selected === i ? 'border-blue-300 bg-blue-50' : 'border-transparent hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${type.color}`} />
              <div className="flex-1">
                <div className="font-semibold text-gray-800">{type.name}</div>
                <div className="text-sm text-gray-500">{type.description}</div>
              </div>
            </div>
            
            {selected === i && (
              <div className="mt-3 pt-3 border-t border-gray-200 grid grid-cols-3 gap-2 text-sm">
                <div>
                  <div className="text-gray-500">示例</div>
                  <div className="text-gray-800">{type.example}</div>
                </div>
                <div>
                  <div className="text-gray-500">内存减少</div>
                  <div className="text-gray-800">{type.memoryReduction}</div>
                </div>
                <div>
                  <div className="text-gray-500">复杂度</div>
                  <div className="text-gray-800">{type.complexity}</div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
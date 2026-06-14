'use client';

import { useState } from 'react';

export function MoEArchitectureDiagram() {
  const [selectedExpert, setSelectedExpert] = useState<number | null>(null);

  const experts = [
    { id: 0, name: '专家1', weight: 0.35, activated: true },
    { id: 1, name: '专家2', weight: 0.25, activated: true },
    { id: 2, name: '专家3', weight: 0.15, activated: false },
    { id: 3, name: '专家4', weight: 0.12, activated: false },
    { id: 4, name: '专家5', weight: 0.08, activated: true },
    { id: 5, name: '专家6', weight: 0.05, activated: false },
  ];

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Mixture of Experts 架构</h2>
      
      <div className="flex flex-col items-center gap-4">
        <div className="p-4 bg-blue-100 rounded-xl text-center w-full">
          <div className="text-sm text-blue-500">输入</div>
          <div className="font-bold text-blue-800">Token Hidden States</div>
        </div>
        
        <div className="text-2xl">↓</div>
        
        <div className="p-4 bg-yellow-100 rounded-xl text-center w-full">
          <div className="text-sm text-yellow-500">路由网络 (Router)</div>
          <div className="font-bold text-yellow-800">Top-K 专家选择</div>
        </div>
        
        <div className="text-2xl">↓</div>
        
        <div className="grid grid-cols-3 gap-2 w-full">
          {experts.map((e) => (
            <div
              key={e.id}
              onClick={() => setSelectedExpert(selectedExpert === e.id ? null : e.id)}
              className={`p-3 rounded-lg cursor-pointer transition-all ${
                e.activated
                  ? 'bg-green-100 border-2 border-green-400'
                  : 'bg-gray-100 border-2 border-transparent opacity-50'
              } ${selectedExpert === e.id ? 'ring-2 ring-blue-400' : ''}`}
            >
              <div className="text-sm font-medium text-gray-700">{e.name}</div>
              <div className="text-xs text-gray-500">权重: {(e.weight * 100).toFixed(0)}%</div>
              {e.activated && <div className="text-xs text-green-600 mt-1">✓ 激活</div>}
            </div>
          ))}
        </div>
        
        <div className="text-2xl">↓</div>
        
        <div className="p-4 bg-purple-100 rounded-xl text-center w-full">
          <div className="text-sm text-purple-500">组合</div>
          <div className="font-bold text-purple-800">加权求和输出</div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 MoE通过稀疏激活实现大模型容量，同时保持推理效率
      </div>
    </div>
  );
}
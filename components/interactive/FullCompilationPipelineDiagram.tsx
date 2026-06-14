'use client';

import { useState } from 'react';

const stages = [
  {
    name: 'Python 前端',
    color: '#6B7280',
    icon: '📝',
    sub: ['TorchScript / ONNX', 'TileLang DSL', 'Relax Function'],
  },
  {
    name: 'IR 生成',
    color: '#3B82F6',
    icon: '📄',
    sub: ['Relax IR', 'TensorIR', 'TIR (low-level)'],
  },
  {
    name: '图优化',
    color: '#8B5CF6',
    icon: '🔧',
    sub: ['常量折叠', '死代码消除', '算子融合', '内存规划'],
  },
  {
    name: '代码生成',
    color: '#F59E0B',
    icon: '⚙️',
    sub: ['TensorIR lowering', 'Tiling', 'Vectorization'],
  },
  {
    name: 'LLVM 后端',
    color: '#10B981',
    icon: '🔩',
    sub: ['LLVM IR', '优化 Pass', '机器码生成'],
  },
  {
    name: '目标代码',
    color: '#EF4444',
    icon: '🎯',
    sub: ['PTX (NVIDIA)', 'HIP (AMD)', 'Ascend C (NPU)', 'Metal (Apple)'],
  },
];

export default function FullCompilationPipelineDiagram() {
  const [activeStage, setActiveStage] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">完整编译管线</h2>
      <p className="text-sm text-gray-400 mb-6">Python → IR → Lowering → LLVM → PTX/HIP/Ascend C</p>

      {/* Pipeline visualization */}
      <div className="flex gap-1 mb-6 overflow-x-auto pb-2">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <button
              className={`flex flex-col items-center min-w-[100px] py-3 px-2 rounded-xl transition-all ${
                i <= activeStage ? 'opacity-100' : 'opacity-30'
              }`}
              style={{
                backgroundColor: i === activeStage ? `${s.color}20` : 'transparent',
                border: `2px solid ${i === activeStage ? s.color : 'transparent'}`,
                boxShadow: i === activeStage ? `0 0 20px ${s.color}30` : 'none',
              }}
              onClick={() => setActiveStage(i)}>
              <div className="text-2xl mb-1">{s.icon}</div>
              <span className="text-[10px] font-bold" style={{ color: s.color }}>{s.name}</span>
            </button>
            {i < stages.length - 1 && (
              <div className="flex flex-col items-center mx-0.5">
                <div className={`w-4 h-0.5 ${i < activeStage ? 'bg-blue-500' : 'bg-gray-700'}`} />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Active stage detail */}
      <div className="bg-gray-800 rounded-xl p-5 mb-4">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-12 h-12 rounded-lg flex items-center justify-center text-2xl"
            style={{ backgroundColor: `${stages[activeStage].color}30` }}>
            {stages[activeStage].icon}
          </div>
          <div>
            <div className="font-bold text-lg" style={{ color: stages[activeStage].color }}>
              Stage {activeStage + 1}: {stages[activeStage].name}
            </div>
            <div className="text-xs text-gray-400">编译管线第 {activeStage + 1} / {stages.length} 阶段</div>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {stages[activeStage].sub.map((item, j) => (
            <div key={j} className="px-3 py-1.5 rounded-full text-xs border"
              style={{ borderColor: `${stages[activeStage].color}60`, color: stages[activeStage].color, backgroundColor: `${stages[activeStage].color}10` }}>
              {item}
            </div>
          ))}
        </div>
      </div>

      {/* Stage summary */}
      <div className="grid grid-cols-6 gap-1">
        {stages.map((s, i) => (
          <div key={i} className={`p-2 rounded text-center text-[10px] transition-all cursor-pointer ${
            i === activeStage ? 'opacity-100' : 'opacity-40'
          }`} style={{ backgroundColor: `${s.color}15`, color: s.color }}
            onClick={() => setActiveStage(i)}>
            <div className="font-bold">{s.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

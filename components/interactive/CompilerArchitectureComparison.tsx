'use client';

import { useState } from 'react';

const architectures = [
  {
    name: 'TileLang',
    layers: [
      { name: 'Python DSL', desc: '高级内核描述', color: 'bg-blue-500' },
      { name: 'TileLang IR', desc: 'TIR + 调度原语', color: 'bg-blue-400' },
      { name: '自动分块', desc: '内存/计算优化', color: 'bg-blue-600' },
      { name: '代码生成', desc: 'CUDA/ROCm', color: 'bg-blue-700' },
    ],
    features: ['自动内存管理', '内置调度策略', '多硬件支持'],
  },
  {
    name: 'TVM',
    layers: [
      { name: 'Relay/Frontend', desc: '模型导入', color: 'bg-green-500' },
      { name: 'TIR', desc: '张量 IR', color: 'bg-green-400' },
      { name: 'AutoSchedule', desc: '自动调优', color: 'bg-green-600' },
      { name: 'Code Generator', desc: 'LLVM/NVVM', color: 'bg-green-700' },
    ],
    features: ['全栈编译', '自动调优', '丰富后端'],
  },
  {
    name: 'XLA',
    layers: [
      { name: 'HLO', desc: '高层优化', color: 'bg-purple-500' },
      { name: 'MLIR', desc: '多级 IR', color: 'bg-purple-400' },
      { name: '优化 Pass', desc: '图优化', color: 'bg-purple-600' },
      { name: 'Backend', desc: 'TPU/GPU', color: 'bg-purple-700' },
    ],
    features: ['深度集成', '自动微分', 'TPU 原生'],
  },
];

export default function CompilerArchitectureComparison() {
  const [selectedArch, setSelectedArch] = useState<number>(0);
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">编译器架构对比</h2>
      <p className="text-gray-400 text-sm mb-4">对比 TileLang vs TVM vs XLA 的编译器架构设计</p>

      <div className="flex gap-2 mb-6">
        {architectures.map((arch, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedArch(idx)}
            className={`px-4 py-2 rounded text-sm font-medium transition-all ${
              selectedArch === idx ? 'bg-blue-600' : 'bg-gray-800 text-gray-400'
            }`}
          >
            {arch.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6 mb-6">
        {architectures.map((arch, aidx) => (
          <div
            key={aidx}
            className={`bg-gray-800 rounded-lg p-4 transition-all ${
              selectedArch === aidx ? 'ring-2 ring-blue-500' : 'opacity-70'
            }`}
            onClick={() => setSelectedArch(aidx)}
          >
            <h3 className="text-sm font-bold text-center mb-4 text-white">{arch.name}</h3>
            <div className="space-y-2">
              {arch.layers.map((layer, lidx) => (
                <div
                  key={lidx}
                  className={`${layer.color} rounded p-3 text-center relative cursor-pointer transition-all ${
                    hoveredLayer === lidx && selectedArch === aidx ? 'scale-105' : ''
                  }`}
                  onMouseEnter={() => setHoveredLayer(lidx)}
                  onMouseLeave={() => setHoveredLayer(null)}
                >
                  <div className="text-xs font-bold text-white">{layer.name}</div>
                  <div className="text-[10px] text-white/70">{layer.desc}</div>
                  {hoveredLayer === lidx && selectedArch === aidx && (
                    <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-800 rotate-45" />
                  )}
                </div>
              ))}
            </div>
            <div className="mt-4 space-y-1">
              {arch.features.map((feat, fidx) => (
                <div key={fidx} className="flex items-center gap-2 text-xs text-gray-400">
                  <span className="text-green-400">✓</span>
                  {feat}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">架构特点对比</h3>
        <div className="grid grid-cols-3 gap-4 text-xs">
          <div>
            <div className="text-blue-400 font-bold mb-1">TileLang</div>
            <div className="text-gray-400 space-y-1">
              <div>• 聚焦内核级优化</div>
              <div>• 自动分块和内存管理</div>
              <div>• 面向 AI 内核场景</div>
            </div>
          </div>
          <div>
            <div className="text-green-400 font-bold mb-1">TVM</div>
            <div className="text-gray-400 space-y-1">
              <div>• 全栈编译框架</div>
              <div>• 自动搜索最优调度</div>
              <div>• 支持多种硬件后端</div>
            </div>
          </div>
          <div>
            <div className="text-purple-400 font-bold mb-1">XLA</div>
            <div className="text-gray-400 space-y-1">
              <div>• 深度框架集成</div>
              <div>• 图级别优化</div>
              <div>• TPU 原生支持</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

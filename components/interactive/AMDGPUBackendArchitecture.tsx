'use client';
import { useState } from 'react';

const pipelineStages = [
  {
    name: 'TileLang IR',
    icon: '📝',
    color: 'border-cyan-500 bg-cyan-900/20',
    textColor: 'text-cyan-300',
    description: 'TileLang领域特定语言中间表示',
    details: ['算子描述', '调度信息', '类型系统'],
  },
  {
    name: '优化Pass',
    icon: '⚙️',
    color: 'border-blue-500 bg-blue-900/20',
    textColor: 'text-blue-300',
    description: 'AMD平台特定优化',
    details: ['MFMA指令选择', 'VGPR分配', 'Wavefront调度'],
  },
  {
    name: 'MLIR',
    icon: '🔧',
    color: 'border-purple-500 bg-purple-900/20',
    textColor: 'text-purple-300',
    description: '多级中间表示',
    details: ['Linalg方言', 'SCF方言', 'ROCDL方言'],
  },
  {
    name: 'LLVM IR',
    icon: '📦',
    color: 'border-green-500 bg-green-900/20',
    textColor: 'text-green-300',
    description: 'LLVM通用中间表示',
    details: ['AMDGPU后端', '寄存器分配', '指令选择'],
  },
  {
    name: 'AMDGPU Code Object',
    icon: '💻',
    color: 'border-orange-500 bg-orange-900/20',
    textColor: 'text-orange-300',
    description: 'ROCm设备代码对象',
    details: ['ELF格式', 'GCN/CDNA指令', '内核元数据'],
  },
  {
    name: 'HIP Runtime',
    icon: '🚀',
    color: 'border-red-500 bg-red-900/20',
    textColor: 'text-red-300',
    description: 'HIP运行时接口',
    details: ['kernel启动', '内存管理', '流同步'],
  },
];

export function AMDGPUBackendArchitecture() {
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-red-400 mb-6">AMD GPU后端架构</h2>

      {/* Pipeline */}
      <div className="flex items-center gap-2 mb-6 overflow-x-auto pb-2">
        {pipelineStages.map((stage, i) => (
          <div key={i} className="flex items-center gap-2">
            <div
              onClick={() => setSelectedStage(selectedStage === i ? null : i)}
              className={`border-2 rounded-xl p-4 cursor-pointer transition-all min-w-[130px] ${
                selectedStage === i ? stage.color + ' ring-2 ring-white/20' : 'border-gray-700 hover:border-gray-500'
              }`}>
              <div className="text-center">
                <div className="text-2xl mb-2">{stage.icon}</div>
                <div className={`font-medium text-xs ${stage.textColor}`}>{stage.name}</div>
              </div>
            </div>
            {i < pipelineStages.length - 1 && (
              <svg className="w-5 h-5 text-gray-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
          </div>
        ))}
      </div>

      {selectedStage !== null && (
        <div className={`p-5 rounded-xl mb-4 border-2 ${pipelineStages[selectedStage].color}`}>
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">{pipelineStages[selectedStage].icon}</span>
            <div>
              <div className={`font-bold ${pipelineStages[selectedStage].textColor}`}>
                {pipelineStages[selectedStage].name}
              </div>
              <div className="text-sm text-gray-400">{pipelineStages[selectedStage].description}</div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {pipelineStages[selectedStage].details.map((d, j) => (
              <div key={j} className="bg-gray-800/50 rounded-lg px-3 py-2 text-xs text-gray-300">
                {d}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AMD vs NVIDIA backend comparison */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-red-400 font-medium mb-2">AMD ROCm栈</div>
          <div className="space-y-1 text-xs text-gray-400 font-mono">
            <div>TileLang IR</div>
            <div className="text-gray-600">↓</div>
            <div>MLIR (ROCDL)</div>
            <div className="text-gray-600">↓</div>
            <div>LLVM AMDGPU</div>
            <div className="text-gray-600">↓</div>
            <div>Code Object (.co)</div>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-400 font-medium mb-2">NVIDIA CUDA栈</div>
          <div className="space-y-1 text-xs text-gray-400 font-mono">
            <div>TileLang IR</div>
            <div className="text-gray-600">↓</div>
            <div>LLVM NVPTX</div>
            <div className="text-gray-600">↓</div>
            <div>PTX / SASS</div>
            <div className="text-gray-600">↓</div>
            <div>cuModule (.cubin)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

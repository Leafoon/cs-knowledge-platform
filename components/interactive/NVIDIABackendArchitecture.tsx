'use client';
import { useState } from 'react';

const pipelineStages = [
  {
    name: 'TileLang IR',
    icon: '📝',
    color: 'border-cyan-500 bg-cyan-900/20',
    textColor: 'text-cyan-300',
    description: 'TileLang领域特定语言的中间表示',
    features: ['算子描述', '调度信息', '类型系统', '优化标注'],
  },
  {
    name: '优化Pass',
    icon: '⚙️',
    color: 'border-blue-500 bg-blue-900/20',
    textColor: 'text-blue-300',
    description: 'IR级别的各种优化变换',
    features: ['算子融合', '内存优化', '循环变换', '向量化'],
  },
  {
    name: 'LLVM IR',
    icon: '🔧',
    color: 'border-purple-500 bg-purple-900/20',
    textColor: 'text-purple-300',
    description: '通用LLVM中间表示',
    features: ['SSA形式', '类型推导', '控制流图', '内联展开'],
  },
  {
    name: 'PTX',
    icon: '💻',
    color: 'border-green-500 bg-green-900/20',
    textColor: 'text-green-300',
    description: 'NVIDIA并行线程执行汇编',
    features: ['虚拟指令集', '寄存器分配', '指令调度', 'warp级操作'],
  },
  {
    name: 'SASS',
    icon: '📦',
    color: 'border-orange-500 bg-orange-900/20',
    textColor: 'text-orange-300',
    description: '目标GPU的机器码',
    features: ['物理寄存器', '硬件指令', 'pipeline调度', '性能计数器'],
  },
  {
    name: 'cuModule',
    icon: '🚀',
    color: 'border-red-500 bg-red-900/20',
    textColor: 'text-red-300',
    description: '可执行的GPU模块',
    features: ['动态加载', 'kernel启动', '内存管理', '多stream'],
  },
];

export function NVIDIABackendArchitecture() {
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">NVIDIA后端架构流水线</h2>

      {/* Pipeline */}
      <div className="flex items-center gap-2 mb-6 overflow-x-auto pb-2">
        {pipelineStages.map((stage, i) => (
          <div key={i} className="flex items-center gap-2">
            <div
              onClick={() => setSelectedStage(selectedStage === i ? null : i)}
              className={`border-2 rounded-xl p-4 cursor-pointer transition-all min-w-[140px] ${
                selectedStage === i ? stage.color + ' ring-2 ring-white/20' : 'border-gray-700 hover:border-gray-500'
              }`}>
              <div className="text-center">
                <div className="text-2xl mb-2">{stage.icon}</div>
                <div className={`font-medium text-sm ${stage.textColor}`}>{stage.name}</div>
              </div>
            </div>
            {i < pipelineStages.length - 1 && (
              <svg className="w-6 h-6 text-gray-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
          </div>
        ))}
      </div>

      {/* Details */}
      {selectedStage !== null && (
        <div className="p-5 bg-gray-800 rounded-xl mb-4">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-2xl">{pipelineStages[selectedStage].icon}</span>
            <div>
              <div className={`font-bold ${pipelineStages[selectedStage].textColor}`}>
                {pipelineStages[selectedStage].name}
              </div>
              <div className="text-sm text-gray-400">{pipelineStages[selectedStage].description}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {pipelineStages[selectedStage].features.map((f, j) => (
              <div key={j} className="bg-gray-700/50 rounded-lg px-3 py-2 text-sm text-gray-300 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />{f}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Architecture summary */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-cyan-400 font-medium mb-1">前端 (Frontend)</div>
          <div className="text-gray-400 text-xs">TileLang IR → 优化Pass → LLVM IR</div>
          <div className="text-gray-500 text-[10px] mt-1">负责语义分析和优化</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-blue-400 font-medium mb-1">后端 (Backend)</div>
          <div className="text-gray-400 text-xs">LLVM IR → PTX → SASS</div>
          <div className="text-gray-500 text-[10px] mt-1">负责目标代码生成</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-400 font-medium mb-1">运行时 (Runtime)</div>
          <div className="text-gray-400 text-xs">cuModule → CUDA Driver API</div>
          <div className="text-gray-500 text-[10px] mt-1">负责模块加载和kernel启动</div>
        </div>
      </div>
    </div>
  );
}

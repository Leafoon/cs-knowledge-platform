'use client';
import { useState } from 'react';

const pipelineStages = [
  {
    name: 'TileLang IR',
    icon: '📝',
    color: 'border-cyan-500 bg-cyan-900/20',
    textColor: 'text-cyan-300',
    description: 'TileLang领域特定语言中间表示',
    details: ['算子描述', '调度信息', '类型系统', 'Cube/Vector标注'],
  },
  {
    name: '昇腾IR',
    icon: '🔧',
    color: 'border-blue-500 bg-blue-900/20',
    textColor: 'text-blue-300',
    description: '昇腾平台专用中间表示',
    details: ['TBE算子描述', '内存布局分析', '数据类型标注', '硬件映射'],
  },
  {
    name: 'Ascend C',
    icon: '💻',
    color: 'border-purple-500 bg-purple-900/20',
    textColor: 'text-purple-300',
    description: '昇腾C++编程语言',
    details: ['Cube/Vector API', 'Pipe流水线', 'TQue数据队列', 'Local Memory管理'],
  },
  {
    name: 'BiSheng编译器',
    icon: '⚙️',
    color: 'border-green-500 bg-green-900/20',
    textColor: 'text-green-300',
    description: '毕昇编译器',
    details: ['IR优化', '指令调度', '寄存器分配', '循环优化'],
  },
  {
    name: 'NPU二进制',
    icon: '📦',
    color: 'border-orange-500 bg-orange-900/20',
    textColor: 'text-orange-300',
    description: 'NPU可执行代码',
    details: ['DVPP指令', 'AI Core指令', '内存分配', 'Kernel启动'],
  },
  {
    name: 'AscendCL运行时',
    icon: '🚀',
    color: 'border-red-500 bg-red-900/20',
    textColor: 'text-red-300',
    description: '昇腾计算语言运行时',
    details: ['Context管理', 'Stream调度', '内存管理', 'Profiling'],
  },
];

export function AscendBackendArchitecture() {
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-orange-400 mb-6">昇腾后端架构流水线</h2>

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
          <div className="grid grid-cols-2 gap-2">
            {pipelineStages[selectedStage].details.map((d, j) => (
              <div key={j} className="bg-gray-800/50 rounded-lg px-3 py-2 text-xs text-gray-300 flex items-center gap-2">
                <span className="text-orange-500">▸</span>{d}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ascend specific: Cube + Vector */}
      <div className="border border-gray-700 rounded-xl p-4 mb-4">
        <div className="text-sm text-gray-400 mb-3 font-medium">Ascend C编程模型</div>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="text-cyan-400 font-medium text-sm mb-2">Cube算子</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>▸ MatMul矩阵乘法</div>
              <div>▸ Conv2d卷积运算</div>
              <div>▸ 利用Cube Core硬件</div>
              <div>▸ 数据从GM→L1→UB搬运</div>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="text-purple-400 font-medium text-sm mb-2">Vector算子</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>▸ ReLU/Softmax激活</div>
              <div>▸ BatchNorm归一化</div>
              <div>▸ 利用Vector Core硬件</div>
              <div>▸ 128-wide SIMD并行</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-cyan-400 font-medium mb-1">前端</div>
          <div className="text-gray-400 text-xs">TileLang → Ascend C</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-purple-400 font-medium mb-1">编译器</div>
          <div className="text-gray-400 text-xs">BiSheng + TBE</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-green-400 font-medium mb-1">运行时</div>
          <div className="text-gray-400 text-xs">AscendCL + CANN</div>
        </div>
      </div>
    </div>
  );
}

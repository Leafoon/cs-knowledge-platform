'use client';

import { useState } from 'react';

const stages = [
  { name: 'Python 解析', color: '#6B7280', icon: '📝', desc: 'AST → Relax IR', detail: '解析 Python 代码生成计算图' },
  { name: '图优化', color: '#3B82F6', icon: '🔧', desc: '常量折叠 / 死代码消除', detail: '多次 Pass 优化计算图' },
  { name: '算子融合', color: '#8B5CF6', icon: '🧩', desc: '连续算子合并为内核', detail: 'Pattern matching + 垂直融合' },
  { name: '内存规划', color: '#F59E0B', icon: '💾', desc: 'Buffer 复用 / 分配策略', detail: '线性扫描内存分配' },
  { name: '代码生成', color: '#10B981', icon: '⚙️', desc: 'TensorIR → 目标代码', detail: 'LLVM/CUDA 后端生成' },
  { name: '运行时执行', color: '#EF4444', icon: '🚀', desc: 'Relax VM 执行', detail: '指令分发 + 算子调度' },
];

export default function EndToEndCompilationFlow() {
  const [activeStage, setActiveStage] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">端到端编译流程</h2>
      <p className="text-sm text-gray-400 mb-6">从 Python 代码到 GPU 执行的完整链路</p>

      <div className="flex flex-col gap-2 mb-6">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center gap-3">
            <button
              className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                i <= activeStage ? 'opacity-100' : 'opacity-40'
              }`}
              style={{
                backgroundColor: i === activeStage ? `${s.color}15` : 'transparent',
                border: `2px solid ${i === activeStage ? s.color : 'transparent'}`,
              }}
              onClick={() => setActiveStage(i)}>
              <div className="w-10 h-10 rounded-full flex items-center justify-center text-lg flex-shrink-0"
                style={{ backgroundColor: `${s.color}30` }}>
                {s.icon}
              </div>
              <div className="text-left flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500">Stage {i + 1}</span>
                  <span className="font-bold text-sm" style={{ color: s.color }}>{s.name}</span>
                </div>
                <div className="text-xs text-gray-400">{s.desc}</div>
              </div>
              {i < stages.length - 1 && (
                <svg width="20" height="20" className="flex-shrink-0">
                  <path d="M10 2 L10 15" stroke={i < activeStage ? s.color : '#4B5563'} strokeWidth="1.5"
                    markerEnd="url(#e2e-arr)" />
                  <defs><marker id="e2e-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
                  </marker></defs>
                </svg>
              )}
            </button>
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: stages[activeStage].color }} />
          <span className="font-bold" style={{ color: stages[activeStage].color }}>
            Stage {activeStage + 1}: {stages[activeStage].name}
          </span>
        </div>
        <div className="text-sm text-gray-300">{stages[activeStage].detail}</div>
      </div>
    </div>
  );
}

'use client';

import React, { useState } from 'react';

interface DiagramItem {
  id: string;
  label: string;
  desc: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  icon: string;
}

const items: DiagramItem[] = [
  { id: 'model', label: '输入模型', desc: 'PyTorch/ONNX/TF 格式的深度学习模型', x: 40, y: 20, w: 120, h: 50, color: 'from-indigo-500 to-indigo-700', icon: '📦' },
  { id: 'relay', label: 'Relay IR', desc: '计算图的函数式中间表示', x: 240, y: 20, w: 120, h: 50, color: 'from-purple-500 to-purple-700', icon: '🔗' },
  { id: 'pass', label: '优化 Pass', desc: '图优化：融合/常量折叠/CSE', x: 440, y: 20, w: 120, h: 50, color: 'from-violet-500 to-violet-700', icon: '🔧' },
  { id: 'tir', label: 'TIR', desc: '包含循环嵌套的底层 IR', x: 240, y: 120, w: 120, h: 50, color: 'from-blue-500 to-blue-700', icon: '📐' },
  { id: 'schedule', label: 'Schedule', desc: '调度优化：tiling/vectorize/parallel', x: 440, y: 120, w: 120, h: 50, color: 'from-cyan-500 to-cyan-700', icon: '📅' },
  { id: 'codegen', label: 'CodeGen', desc: '生成目标设备代码', x: 440, y: 220, w: 120, h: 50, color: 'from-emerald-500 to-emerald-700', icon: '⚙️' },
  { id: 'runtime', label: 'Runtime', desc: '在设备上执行编译后的模型', x: 240, y: 220, w: 120, h: 50, color: 'from-teal-500 to-teal-700', icon: '🚀' },
];

const arrows = [
  { from: 'model', to: 'relay' },
  { from: 'relay', to: 'pass' },
  { from: 'pass', to: 'tir' },
  { from: 'tir', to: 'schedule' },
  { from: 'schedule', to: 'codegen' },
  { from: 'codegen', to: 'runtime' },
];

export function PracticeExercisesDiagram() {
  const [active, setActive] = useState<string | null>(null);
  const [step, setStep] = useState(0);

  const getItem = (id: string) => items.find((i) => i.id === id)!;

  return (
    <div className="w-full rounded-xl border border-white/10 bg-gradient-to-br from-gray-900 via-gray-950 to-black p-6">
      <h3 className="mb-2 text-lg font-bold text-white">练习题图示说明</h3>
      <p className="mb-4 text-sm text-gray-400">
        通过图示理解 TVM 编译流程中各阶段的关系，逐步点击下方按钮查看每一步。
      </p>
      <svg viewBox="0 0 620 290" className="w-full">
        {arrows.map((a, i) => {
          const from = getItem(a.from);
          const to = getItem(a.to);
          const isHighlighted = i < step;
          return (
            <line
              key={i}
              x1={from.x + from.w / 2} y1={from.y + from.h / 2}
              x2={to.x + to.w / 2} y2={to.y + to.h / 2}
              stroke={isHighlighted ? '#818cf8' : '#374151'}
              strokeWidth={isHighlighted ? 2 : 1}
              strokeDasharray={isHighlighted ? '' : '6 4'}
              markerEnd="url(#arrow2)"
            />
          );
        })}
        <defs>
          <marker id="arrow2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
          </marker>
        </defs>
        {items.map((item, i) => {
          const isVisible = i < step;
          const isActive = active === item.id;
          return (
            <g
              key={item.id}
              onMouseEnter={() => setActive(item.id)}
              onMouseLeave={() => setActive(null)}
              className="cursor-pointer"
              opacity={isVisible ? 1 : 0.3}
            >
              <rect
                x={item.x} y={item.y}
                width={item.w} height={item.h}
                rx={10}
                fill={isActive ? '#4f46e5' : '#1e1b4b'}
                stroke={isActive ? '#a5b4fc' : '#312e81'}
                strokeWidth={isActive ? 2 : 1}
              />
              <text x={item.x + item.w / 2} y={item.y + 20} textAnchor="middle" className="text-xs" fill="white">
                {item.icon} {item.label}
              </text>
              <text x={item.x + item.w / 2} y={item.y + 36} textAnchor="middle" className="text-[9px]" fill="#a5b4fc">
                {item.desc}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-4 flex items-center justify-center gap-3">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="rounded-lg bg-white/5 px-3 py-1.5 text-xs text-gray-300 hover:bg-white/10 disabled:opacity-30"
        >
          ← 上一步
        </button>
        <span className="text-xs text-gray-400">步骤 {step}/{items.length}</span>
        <button
          onClick={() => setStep(Math.min(items.length, step + 1))}
          disabled={step === items.length}
          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs text-white hover:bg-indigo-500 disabled:opacity-30"
        >
          下一步 →
        </button>
      </div>
      {active && (
        <div className="mt-3 text-center text-xs text-indigo-300">
          {items.find((i) => i.id === active)?.desc}
        </div>
      )}
    </div>
  );
}

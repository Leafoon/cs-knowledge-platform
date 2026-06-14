'use client';

import { useState } from 'react';

const stages = [
  { name: 'Prologue', color: '#60A5FA', desc: '加载全局内存到共享内存', instructions: ['ld.global', 'bar.sync', 'st.shared'] },
  { name: 'Main Loop', color: '#34D399', desc: '计算核心：矩阵乘累加', instructions: ['ld.shared', 'mma.sync', 'bar.sync', 'st.shared'] },
  { name: 'Epilogue', color: '#F97316', desc: '将结果写回全局内存', instructions: ['st.global'] },
];

const timeline = [
  { cycle: 0, stage: 0, inst: 'ld.global' },
  { cycle: 1, stage: 0, inst: 'bar.sync' },
  { cycle: 2, stage: 0, inst: 'st.shared' },
  { cycle: 3, stage: 1, inst: 'ld.shared' },
  { cycle: 4, stage: 1, inst: 'mma.sync' },
  { cycle: 5, stage: 1, inst: 'bar.sync' },
  { cycle: 6, stage: 1, inst: 'ld.shared' },
  { cycle: 7, stage: 1, inst: 'mma.sync' },
  { cycle: 8, stage: 2, inst: 'st.global' },
];

export default function PipelineStageVisualizer() {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">流水线阶段可视化</h2>

      <div className="flex gap-4 mb-6">
        {stages.map((s, i) => (
          <div
            key={i}
            className="flex-1 p-4 rounded-lg border-2 cursor-pointer transition-all"
            style={{ borderColor: activeIdx === i ? s.color : 'transparent', backgroundColor: `${s.color}22` }}
            onClick={() => setActiveIdx(activeIdx === i ? null : i)}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: s.color }} />
              <span className="font-bold">{s.name}</span>
            </div>
            <p className="text-sm text-gray-300 mb-2">{s.desc}</p>
            {activeIdx === i && (
              <div className="mt-2 space-y-1">
                {s.instructions.map((inst, j) => (
                  <div key={j} className="text-xs bg-black/40 rounded px-2 py-1 font-mono">{inst}</div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="overflow-x-auto">
        <svg viewBox="0 0 900 120" className="w-full h-28">
          <text x="0" y="15" fill="#9CA3AF" fontSize="11">周期</text>
          {timeline.map((t, i) => (
            <g key={i}>
              <text x={i * 100 + 5} y="15" fill="#9CA3AF" fontSize="10">{t.cycle}</text>
              <rect
                x={i * 100 + 5}
                y="25"
                width="85"
                height="40"
                rx="6"
                fill={stages[t.stage].color}
                opacity={0.7}
              />
              <text x={i * 100 + 47} y="50" fill="white" fontSize="10" textAnchor="middle">{t.inst}</text>
              <text x={i * 100 + 47} y="80" fill="#9CA3AF" fontSize="9" textAnchor="middle">{stages[t.stage].name}</text>
              {i < timeline.length - 1 && (
                <line x1={i * 100 + 90} y1="45" x2={(i + 1) * 100 + 5} y2="45" stroke="#4B5563" strokeWidth="1.5" markerEnd="url(#arrow)" />
              )}
            </g>
          ))}
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
            </marker>
          </defs>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        {['LD→共享内存', '共享内存计算', 'ST→全局内存'].map((label, i) => (
          <div key={i} className="bg-gray-800 rounded p-2 text-center">
            <div className="font-mono text-gray-400">阶段 {i + 1}</div>
            <div>{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

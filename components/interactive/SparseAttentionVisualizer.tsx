'use client';

import { useState } from 'react';

const patterns = [
  {
    name: 'Dense', desc: '全注意力，每个 token 关注所有 token',
    size: 6, pattern: Array.from({ length: 6 }, () => Array(6).fill(true)),
  },
  {
    name: 'Sliding Window', desc: '窗口大小为 3，每个 token 仅关注相邻 token',
    size: 6, pattern: Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 6 }, (_, j) => Math.abs(i - j) <= 1)),
  },
  {
    name: 'Sparse (Strided)', desc: '每 3 个 token 选择一个',
    size: 6, pattern: Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 6 }, (_, j) => i % 3 === j % 3)),
  },
  {
    name: 'Grouped', desc: '4 个 head，每组独立注意力',
    size: 6, pattern: Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 6 }, (_, j) => Math.floor(i / 1.5) === Math.floor(j / 1.5))),
  },
];

export default function SparseAttentionVisualizer() {
  const [selectedPattern, setSelectedPattern] = useState(0);

  const { pattern, size } = patterns[selectedPattern];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">注意力模式可视化</h2>

      <div className="flex gap-2 mb-4 flex-wrap">
        {patterns.map((p, i) => (
          <button key={i} onClick={() => setSelectedPattern(i)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
              selectedPattern === i ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}>
            {p.name}
          </button>
        ))}
      </div>

      <p className="text-sm text-gray-400 mb-4">{patterns[selectedPattern].desc}</p>

      <div className="flex gap-6 items-start mb-4">
        <div>
          <div className="text-xs text-gray-400 mb-2 text-center">注意力掩码</div>
          <svg viewBox={`0 0 ${size * 32 + 50} ${size * 32 + 50}`} className="w-64 h-64">
            {Array.from({ length: size }, (_, i) => (
              <g key={i}>
                <text x={i * 32 + 28} y={14} fill="#9CA3AF" fontSize="9" textAnchor="middle">K{i}</text>
                <text x={8} y={i * 32 + 32} fill="#9CA3AF" fontSize="9" textAnchor="middle">Q{i}</text>
                {Array.from({ length: size }, (_, j) => (
                  <rect key={j} x={j * 32 + 18} y={i * 32 + 20} width={28} height={28} rx={3}
                    fill={pattern[i][j] ? '#8B5CF640' : '#1F2937'}
                    stroke={pattern[i][j] ? '#8B5CF6' : '#374151'} strokeWidth={pattern[i][j] ? 1.5 : 0.5} />
                ))}
              </g>
            ))}
          </svg>
        </div>

        <div className="flex-1 space-y-2">
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <div className="font-bold text-purple-400 mb-1">模式统计</div>
            <div className="text-gray-400">
              连接数: {pattern.flat().filter(Boolean).length} / {size * size}
            </div>
            <div className="text-gray-400">
              稀疏率: {((1 - pattern.flat().filter(Boolean).length / (size * size)) * 100).toFixed(0)}%
            </div>
            <div className="text-gray-400">
              复杂度: O({pattern.flat().filter(Boolean).length}) vs O({size * size})
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <div className="font-bold text-blue-400 mb-1">适用场景</div>
            <div className="text-gray-400">
              {selectedPattern === 0 && '小序列、高质量注意力'}
              {selectedPattern === 1 && '长序列建模（Mistral）'}
              {selectedPattern === 2 && '大模型高效推理'}
              {selectedPattern === 3 && 'GQA/MQA 降低 KV 开销'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

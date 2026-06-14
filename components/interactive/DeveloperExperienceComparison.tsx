'use client';

import { useState } from 'react';

const frameworks = [
  { name: 'TileLang', color: 'text-blue-400', bgColor: 'bg-blue-500' },
  { name: 'Triton', color: 'text-green-400', bgColor: 'bg-green-500' },
  { name: 'CUDA', color: 'text-red-400', bgColor: 'bg-red-500' },
];

const dimensions = [
  { name: '学习曲线', scores: [7, 8, 4], desc: '入门难度和上手时间' },
  { name: '调试体验', scores: [9, 6, 5], desc: '问题定位和修复效率' },
  { name: '迭代速度', scores: [8, 7, 5], desc: '从想法到可运行代码' },
  { name: '文档质量', scores: [7, 9, 6], desc: '文档完整性和可读性' },
  { name: '社区支持', scores: [6, 9, 8], desc: '社区活跃度和问题解答' },
  { name: '工具链', scores: [8, 7, 9], desc: '配套工具和 IDE 支持' },
];

export default function DeveloperExperienceComparison() {
  const [hoveredDim, setHoveredDim] = useState<number | null>(null);
  const [selectedFramework, setSelectedFramework] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">开发者体验对比</h2>
      <p className="text-gray-400 text-sm mb-4">从多个维度对比 TileLang、Triton 和 CUDA 的开发体验</p>

      <div className="flex gap-3 mb-6">
        {frameworks.map((fw, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedFramework(selectedFramework === idx ? null : idx)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium transition-all ${
              selectedFramework === idx
                ? `${fw.bgColor} text-white`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${fw.bgColor}`} />
            {fw.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6 mb-6">
        <div className="col-span-2">
          <div className="space-y-3">
            {dimensions.map((dim, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg transition-all ${
                  hoveredDim === idx ? 'bg-gray-800' : 'bg-gray-800/50'
                }`}
                onMouseEnter={() => setHoveredDim(idx)}
                onMouseLeave={() => setHoveredDim(null)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <span className="text-sm font-medium text-white">{dim.name}</span>
                    <span className="text-xs text-gray-500 ml-2">{dim.desc}</span>
                  </div>
                </div>
                <div className="flex gap-4">
                  {frameworks.map((fw, fidx) => (
                    <div key={fidx} className="flex-1">
                      <div className="flex items-center gap-2">
                        <div className="w-24 text-xs text-gray-400">{fw.name}</div>
                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${fw.bgColor} transition-all`}
                            style={{ width: `${dim.scores[fidx] * 10}%` }}
                          />
                        </div>
                        <span className={`text-xs font-bold ${fw.color}`}>
                          {dim.scores[fidx]}/10
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">综合得分</h3>
            {frameworks.map((fw, idx) => {
              const total = dimensions.reduce((sum, dim) => sum + dim.scores[idx], 0);
              const max = dimensions.length * 10;
              return (
                <div key={idx} className="mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className={fw.color}>{fw.name}</span>
                    <span className="text-gray-400">{total}/{max}</span>
                  </div>
                  <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${fw.bgColor}`}
                      style={{ width: `${(total / max) * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-2">关键洞察</h3>
            <div className="space-y-2 text-xs text-gray-400">
              <div className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">●</span>
                <span>TileLang 在调试体验和迭代速度上领先</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">●</span>
                <span>Triton 的文档和社区支持最成熟</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-400 mt-0.5">●</span>
                <span>CUDA 工具链最完善但学习成本最高</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

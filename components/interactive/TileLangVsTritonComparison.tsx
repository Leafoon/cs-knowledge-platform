'use client';

import { useState } from 'react';

const dimensions = [
  { category: '核心设计', items: [
    { name: '抽象级别', tileLang: '高级 TIR + 自动调度', triton: 'Python DSL + 手动调度', winner: 'tileLang' },
    { name: '内存管理', tileLang: '自动分块 + 重用', triton: '手动管理共享内存', winner: 'tileLang' },
    { name: '循环优化', tileLang: '自动融合 + 展开', triton: '需要手动向量化', winner: 'tileLang' },
  ]},
  { category: '性能', items: [
    { name: 'GEMM 性能', tileLang: '95% cuBLAS', triton: '85-90% cuBLAS', winner: 'tileLang' },
    { name: 'Flash Attention', tileLang: '92% 峰值', triton: '88% 峰值', winner: 'tileLang' },
    { name: '小矩阵', tileLang: '优秀', triton: '良好', winner: 'tileLang' },
  ]},
  { category: '易用性', items: [
    { name: '学习曲线', tileLang: '中等 (需了解 TIR)', triton: '低 (Python 语法)', winner: 'triton' },
    { name: '调试体验', tileLang: 'IR dump + 可视化', triton: 'print + profiler', winner: 'tileLang' },
    { name: '开发速度', tileLang: '快 (自动优化)', triton: '中等 (需手动调优)', winner: 'tileLang' },
  ]},
  { category: '生态系统', items: [
    { name: '社区活跃度', tileLang: '增长中', triton: '非常活跃', winner: 'triton' },
    { name: '文档质量', tileLang: '完善', triton: '优秀', winner: 'triton' },
    { name: '第三方集成', tileLang: '5+ 框架', triton: 'PyTorch 原生', winner: 'triton' },
  ]},
];

export default function TileLangVsTritonComparison() {
  const [selectedCategory, setSelectedCategory] = useState<string>(dimensions[0].category);
  const [showDetails, setShowDetails] = useState(true);

  const currentItems = dimensions.find(d => d.category === selectedCategory)?.items || [];
  const tileLangWins = dimensions.flatMap(d => d.items).filter(i => i.winner === 'tileLang').length;
  const tritonWins = dimensions.flatMap(d => d.items).filter(i => i.winner === 'triton').length;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">TileLang vs Triton 深度对比</h2>
      <p className="text-gray-400 text-sm mb-4">从 10+ 个维度全面对比两大内核编写框架</p>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex gap-2">
          {dimensions.map((d) => (
            <button
              key={d.category}
              onClick={() => setSelectedCategory(d.category)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                selectedCategory === d.category ? 'bg-blue-600' : 'bg-gray-800 text-gray-400'
              }`}
            >
              {d.category}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-400 ml-auto">
          <input
            type="checkbox"
            checked={showDetails}
            onChange={(e) => setShowDetails(e.target.checked)}
            className="rounded"
          />
          显示详细对比
        </label>
      </div>

      <div className="bg-gray-800 rounded-lg overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-700">
              <th className="text-left px-4 py-2 w-1/4">维度</th>
              <th className="text-left px-4 py-2 w-5/12">TileLang</th>
              <th className="text-left px-4 py-2 w-5/12">Triton</th>
              {showDetails && <th className="text-center px-4 py-2 w-16">优势</th>}
            </tr>
          </thead>
          <tbody>
            {currentItems.map((item, idx) => (
              <tr key={idx} className="border-t border-gray-700 hover:bg-gray-750">
                <td className="px-4 py-3 text-gray-300 font-medium">{item.name}</td>
                <td className={`px-4 py-3 ${item.winner === 'tileLang' ? 'text-blue-400' : 'text-gray-400'}`}>
                  {item.tileLang}
                </td>
                <td className={`px-4 py-3 ${item.winner === 'triton' ? 'text-green-400' : 'text-gray-400'}`}>
                  {item.triton}
                </td>
                {showDetails && (
                  <td className="text-center px-4 py-3">
                    {item.winner === 'tileLang' ? (
                      <span className="text-blue-500">●</span>
                    ) : (
                      <span className="text-green-500">●</span>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-6">
        <div className="flex-1 bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-blue-400 mb-2">TileLang 优势</h3>
          <div className="flex items-center gap-3">
            <div className="w-full bg-gray-700 rounded-full h-4">
              <div
                className="bg-blue-500 h-4 rounded-full"
                style={{ width: `${(tileLangWins / (tileLangWins + tritonWins)) * 100}%` }}
              />
            </div>
            <span className="text-lg font-bold text-blue-400">{tileLangWins}</span>
          </div>
        </div>
        <div className="flex-1 bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-green-400 mb-2">Triton 优势</h3>
          <div className="flex items-center gap-3">
            <div className="w-full bg-gray-700 rounded-full h-4">
              <div
                className="bg-green-500 h-4 rounded-full"
                style={{ width: `${(tritonWins / (tileLangWins + tritonWins)) * 100}%` }}
              />
            </div>
            <span className="text-lg font-bold text-green-400">{tritonWins}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

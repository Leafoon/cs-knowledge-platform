'use client';

import React, { useState } from 'react';

interface ComparisonRow {
  dimension: string;
  tilelang: string;
  triton: string;
  cuda: string;
  winner: 'tilelang' | 'triton' | 'cuda';
}

export function TileLangVsTritonVsCUDAComparison() {
  const [sortBy, setSortBy] = useState<string>('dimension');

  const comparisons: ComparisonRow[] = [
    { dimension: '抽象级别', tilelang: '高级 (三级接口)', triton: '中级', cuda: '低级', winner: 'tilelang' },
    { dimension: '内存管理', tilelang: '自动管理', triton: '自动管理', cuda: '手动管理', winner: 'tilelang' },
    { dimension: '性能', tilelang: '接近手写', triton: '接近手写', cuda: '最优', winner: 'cuda' },
    { dimension: '学习曲线', tilelang: '平缓', triton: '中等', cuda: '陡峭', winner: 'tilelang' },
    { dimension: '硬件支持', tilelang: 'NVIDIA/AMD/昇腾', triton: 'NVIDIA', cuda: 'NVIDIA', winner: 'tilelang' },
    { dimension: '自动调优', tilelang: '支持', triton: '部分支持', cuda: '不支持', winner: 'tilelang' },
    { dimension: '社区生态', tilelang: '成长中', triton: '活跃', cuda: '成熟', winner: 'cuda' },
    { dimension: '开发效率', tilelang: '高', triton: '中等', cuda: '低', winner: 'tilelang' },
    { dimension: '调试支持', tilelang: '良好', triton: '一般', cuda: '完善', winner: 'cuda' },
    { dimension: '文档质量', tilelang: '完善', triton: '良好', cuda: '完善', winner: 'cuda' },
  ];

  const getWinnerColor = (winner: string) => {
    switch (winner) {
      case 'tilelang': return 'text-green-400';
      case 'triton': return 'text-yellow-400';
      case 'cuda': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">TileLang vs Triton vs CUDA 对比</h2>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left p-3 text-gray-300 font-bold">维度</th>
              <th className="text-left p-3 text-green-400 font-bold">TileLang</th>
              <th className="text-left p-3 text-yellow-400 font-bold">Triton</th>
              <th className="text-left p-3 text-blue-400 font-bold">CUDA</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((row, i) => (
              <tr
                key={i}
                className="border-b border-gray-800 hover:bg-gray-800 transition-colors"
              >
                <td className="p-3 text-white font-medium">{row.dimension}</td>
                <td className={`p-3 ${row.winner === 'tilelang' ? 'text-green-400 font-bold' : 'text-gray-300'}`}>
                  {row.tilelang}
                  {row.winner === 'tilelang' && <span className="ml-2 text-xs">✓</span>}
                </td>
                <td className={`p-3 ${row.winner === 'triton' ? 'text-yellow-400 font-bold' : 'text-gray-300'}`}>
                  {row.triton}
                  {row.winner === 'triton' && <span className="ml-2 text-xs">✓</span>}
                </td>
                <td className={`p-3 ${row.winner === 'cuda' ? 'text-blue-400 font-bold' : 'text-gray-300'}`}>
                  {row.cuda}
                  {row.winner === 'cuda' && <span className="ml-2 text-xs">✓</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-green-900/30 rounded-lg p-4 text-center">
          <div className="text-green-400 text-2xl font-bold">6</div>
          <div className="text-gray-300 text-sm">TileLang 优势项</div>
        </div>
        <div className="bg-yellow-900/30 rounded-lg p-4 text-center">
          <div className="text-yellow-400 text-2xl font-bold">0</div>
          <div className="text-gray-300 text-sm">Triton 优势项</div>
        </div>
        <div className="bg-blue-900/30 rounded-lg p-4 text-center">
          <div className="text-blue-400 text-2xl font-bold">4</div>
          <div className="text-gray-300 text-sm">CUDA 优势项</div>
        </div>
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';

const generatePattern = () => {
  const pattern: number[][] = [];
  for (let i = 0; i < 8; i++) {
    const row: number[] = [];
    for (let j = 0; j < 8; j++) {
      row.push(0);
    }
    for (let k = 0; k < 4; k++) {
      const pos = Math.floor(Math.random() * 4);
      row[pos * 2] = 1;
      row[pos * 2 + 1] = Math.random() > 0.5 ? 1 : 0;
    }
    pattern.push(row);
  }
  return pattern;
};

export default function StructuredSparsityDemo() {
  const [pattern, setPattern] = useState<number[][]>(generatePattern);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [animationStep, setAnimationStep] = useState(0);

  const regenerate = () => {
    setPattern(generatePattern());
    setSelectedCell(null);
    setAnimationStep(0);
  };

  const totalOnes = pattern.flat().filter(x => x === 1).length;
  const totalCells = pattern.length * pattern[0].length;
  const sparsity = ((1 - totalOnes / totalCells) * 100).toFixed(1);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">2:4 结构化稀疏演示</h2>
      <p className="text-gray-400 text-sm mb-4">交互式演示 2:4 结构化稀疏模式 - 每 4 个元素中恰好有 2 个非零</p>

      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={regenerate}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-all"
        >
          重新生成模式
        </button>
        <button
          onClick={() => setAnimationStep(prev => (prev + 1) % 3)}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-all"
        >
          切换视图
        </button>
        <div className="ml-auto flex gap-4 text-xs">
          <span className="text-green-400">非零: {totalOnes}</span>
          <span className="text-gray-400">零: {totalCells - totalOnes}</span>
          <span className="text-blue-400">稀疏率: {sparsity}%</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-sm font-bold text-gray-300 mb-4">2:4 稀疏模式</h3>
          <div className="grid grid-cols-8 gap-1 max-w-[320px] mx-auto">
            {pattern.flat().map((val, idx) => {
              const row = Math.floor(idx / 8);
              const col = idx % 8;
              const isSelected = selectedCell?.row === row && selectedCell?.col === col;
              const inGroup = Math.floor(col / 2) === Math.floor((col + 1) / 2);

              return (
                <div
                  key={idx}
                  className={`w-9 h-9 rounded flex items-center justify-center text-sm font-bold cursor-pointer transition-all ${
                    val === 1
                      ? isSelected
                        ? 'bg-green-400 text-black ring-2 ring-white'
                        : 'bg-green-600 text-white hover:bg-green-500'
                      : isSelected
                      ? 'bg-gray-500 text-white ring-2 ring-white'
                      : 'bg-gray-700 text-gray-500 hover:bg-gray-600'
                  } ${animationStep === 1 && inGroup ? 'ring-1 ring-blue-400' : ''}`}
                  onClick={() => setSelectedCell({ row, col })}
                >
                  {val === 1 ? '1' : '0'}
                </div>
              );
            })}
          </div>
          <div className="flex justify-center gap-4 mt-4 text-xs">
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 bg-green-600 rounded" />
              <span className="text-gray-400">非零 (1)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 bg-gray-700 rounded" />
              <span className="text-gray-400">零 (0)</span>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {selectedCell && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold text-gray-300 mb-3">选中元素信息</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">位置</span>
                  <span className="text-white">({selectedCell.row}, {selectedCell.col})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">值</span>
                  <span className={pattern[selectedCell.row][selectedCell.col] === 1 ? 'text-green-400' : 'text-gray-500'}>
                    {pattern[selectedCell.row][selectedCell.col]}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">2:4 组</span>
                  <span className="text-blue-400">
                    第 {Math.floor(selectedCell.col / 2) + 1} 组
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">组内非零数</span>
                  <span className="text-white">
                    {pattern[selectedCell.row]
                      .slice(Math.floor(selectedCell.col / 2) * 2, Math.floor(selectedCell.col / 2) * 2 + 2)
                      .filter(x => x === 1).length +
                     pattern[selectedCell.row]
                      .slice(Math.floor(selectedCell.col / 2) * 2 + 2, Math.floor(selectedCell.col / 2) * 2 + 4)
                      .filter(x => x === 1).length}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">硬件加速原理</h3>
            <div className="space-y-3 text-xs text-gray-400">
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">1.</span>
                <span>每 4 个元素中选择 2 个非零，形成固定模式</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">2.</span>
                <span>硬件使用 Metadata 存储压缩索引 (2 bits/元素)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">3.</span>
                <span>NVIDIA Ampere+ 的 Sparse Tensor Core 原生支持</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-0.5">4.</span>
                <span>相比密集计算可获得 2x 加速 (理论峰值)</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">内存节省</h3>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-xs text-gray-400 mb-1">密集存储</div>
                <div className="h-4 bg-gray-700 rounded">
                  <div className="h-full bg-gray-500 rounded" style={{ width: '100%' }} />
                </div>
                <div className="text-xs text-gray-500 mt-1">{totalCells * 2} bytes</div>
              </div>
              <div className="flex-1">
                <div className="text-xs text-gray-400 mb-1">稀疏存储</div>
                <div className="h-4 bg-gray-700 rounded">
                  <div
                    className="h-full bg-green-500 rounded"
                    style={{ width: `${(totalOnes / totalCells) * 100 + 10}%` }}
                  />
                </div>
                <div className="text-xs text-green-400 mt-1">
                  {(totalOnes * 2 + totalCells / 4 * 0.5).toFixed(0)} bytes (+metadata)
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

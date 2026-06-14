'use client';

import { useState } from 'react';

const blockSize = 4;
const matrixSize = 16;

const generateSparseMatrix = (density: number) => {
  const blocks: { row: number; col: number; filled: boolean }[] = [];
  for (let r = 0; r < matrixSize / blockSize; r++) {
    for (let c = 0; c < matrixSize / blockSize; c++) {
      blocks.push({
        row: r,
        col: c,
        filled: Math.random() < density,
      });
    }
  }
  return blocks;
};

export default function BlockSparseVisualizer() {
  const [density, setDensity] = useState(0.3);
  const [blocks, setBlocks] = useState(() => generateSparseMatrix(0.3));
  const [selectedBlock, setSelectedBlock] = useState<{ row: number; col: number } | null>(null);

  const regenerate = () => {
    setBlocks(generateSparseMatrix(density));
    setSelectedBlock(null);
  };

  const nonZeroBlocks = blocks.filter(b => b.filled).length;
  const totalBlocks = blocks.length;
  const compressionRatio = ((1 - nonZeroBlocks / totalBlocks) * 100).toFixed(1);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">块稀疏矩阵可视化</h2>
      <p className="text-gray-400 text-sm mb-4">可视化块稀疏矩阵的非零块分布</p>

      <div className="flex items-center gap-6 mb-6">
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-300">稀疏密度:</label>
          <input
            type="range"
            min="0.1"
            max="0.8"
            step="0.05"
            value={density}
            onChange={(e) => {
              setDensity(parseFloat(e.target.value));
              setBlocks(generateSparseMatrix(parseFloat(e.target.value)));
            }}
            className="w-32"
          />
          <span className="text-sm text-gray-400 w-12">{(density * 100).toFixed(0)}%</span>
        </div>
        <button
          onClick={regenerate}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-all"
        >
          重新生成
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-sm font-bold text-gray-300 mb-4">块稀疏矩阵</h3>
          <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${matrixSize / blockSize}, 1fr)` }}>
            {blocks.map((block, idx) => {
              const isSelected = selectedBlock?.row === block.row && selectedBlock?.col === block.col;
              return (
                <div
                  key={idx}
                  className={`aspect-square rounded cursor-pointer transition-all ${
                    block.filled
                      ? isSelected
                        ? 'bg-blue-400 ring-2 ring-white'
                        : 'bg-blue-600 hover:bg-blue-500'
                      : isSelected
                      ? 'bg-gray-600 ring-2 ring-white'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                  onClick={() => setSelectedBlock({ row: block.row, col: block.col })}
                >
                  {block.filled && (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="w-3/4 h-3/4 bg-blue-300/30 rounded grid grid-cols-2 gap-px p-px">
                        {Array.from({ length: blockSize * blockSize }).map((_, i) => (
                          <div key={i} className="bg-blue-400/50 rounded-sm" />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">统计信息</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-400">总块数</div>
                <div className="text-lg font-bold text-white">{totalBlocks}</div>
              </div>
              <div>
                <div className="text-xs text-gray-400">非零块</div>
                <div className="text-lg font-bold text-blue-400">{nonZeroBlocks}</div>
              </div>
              <div>
                <div className="text-xs text-gray-400">零块</div>
                <div className="text-lg font-bold text-gray-400">{totalBlocks - nonZeroBlocks}</div>
              </div>
              <div>
                <div className="text-xs text-gray-400">压缩率</div>
                <div className="text-lg font-bold text-green-400">{compressionRatio}%</div>
              </div>
            </div>
          </div>

          {selectedBlock && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm font-bold text-gray-300 mb-3">选中块信息</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">位置</span>
                  <span className="text-white">({selectedBlock.row}, {selectedBlock.col})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">全局范围</span>
                  <span className="text-white">
                    [{selectedBlock.row * blockSize}:{(selectedBlock.row + 1) * blockSize},
                    {selectedBlock.col * blockSize}:{(selectedBlock.col + 1) * blockSize}]
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">状态</span>
                  <span className={blocks.find(b => b.row === selectedBlock.row && b.col === selectedBlock.col)?.filled ? 'text-green-400' : 'text-gray-500'}>
                    {blocks.find(b => b.row === selectedBlock.row && b.col === selectedBlock.col)?.filled ? '非零块' : '零块'}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">内存节省</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">密集存储</span>
                <span className="text-red-400">{matrixSize * matrixSize * 2} bytes</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">块稀疏存储</span>
                <span className="text-green-400">
                  {(nonZeroBlocks * blockSize * blockSize * 2 + nonZeroBlocks * 4).toFixed(0)} bytes
                </span>
              </div>
              <div className="w-full h-2 bg-gray-700 rounded-full mt-2">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{ width: `${100 - parseFloat(compressionRatio)}%` }}
                />
              </div>
              <div className="text-xs text-gray-400 text-center">
                节省 {compressionRatio}% 存储空间
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

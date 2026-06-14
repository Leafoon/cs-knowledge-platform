'use client';

import React, { useState } from 'react';

interface DecisionNode {
  question: string;
  options: { label: string; next: number | string }[];
}

export function TileShapeSelectionGuide() {
  const [currentNode, setCurrentNode] = useState(0);
  const [path, setPath] = useState<number[]>([]);

  const decisions: DecisionNode[] = [
    {
      question: '你的矩阵规模是？',
      options: [
        { label: '小矩阵 (<512)', next: 1 },
        { label: '中等矩阵 (512-2048)', next: 2 },
        { label: '大矩阵 (>2048)', next: 3 },
      ],
    },
    {
      question: '小矩阵场景，你更关注？',
      options: [
        { label: '低延迟', next: 'tile_32x32' },
        { label: '高吞吐', next: 'tile_64x64' },
      ],
    },
    {
      question: '中等矩阵，你的GPU显存是？',
      options: [
        { label: '≤16GB (如A100)', next: 4 },
        { label: '>16GB (如H100)', next: 5 },
      ],
    },
    {
      question: '大矩阵，你是否需要流水线？',
      options: [
        { label: '是', next: 'tile_128x128_pipeline' },
        { label: '否', next: 'tile_128x128' },
      ],
    },
    {
      question: '小显存GPU，矩阵是否适合SRAM？',
      options: [
        { label: '是', next: 'tile_64x64' },
        { label: '否', next: 'tile_128x64' },
      ],
    },
    {
      question: '大显存GPU，是否使用Tensor Core？',
      options: [
        { label: '是', next: 'tile_128x128_tensor' },
        { label: '否', next: 'tile_64x64' },
      ],
    },
  ];

  const recommendations: Record<string, { tile: string; config: string; performance: string }> = {
    tile_32x32: { tile: '32×32', config: 'BM=32, BN=32, BK=32', performance: '~35 TFLOPS' },
    tile_64x64: { tile: '64×64', config: 'BM=64, BN=64, BK=32', performance: '~65 TFLOPS' },
    tile_128x64: { tile: '128×64', config: 'BM=128, BN=64, BK=32', performance: '~75 TFLOPS' },
    tile_128x128: { tile: '128×128', config: 'BM=128, BN=128, BK=32', performance: '~85 TFLOPS' },
    tile_128x128_pipeline: { tile: '128×128 + Pipeline', config: 'BM=128, BN=128, BK=32, Stages=4', performance: '~90 TFLOPS' },
    tile_128x128_tensor: { tile: '128×128 + Tensor Core', config: 'BM=128, BN=128, BK=32, TC=FP16', performance: '~95 TFLOPS' },
  };

  const handleOption = (next: number | string) => {
    if (typeof next === 'number') {
      setPath([...path, currentNode]);
      setCurrentNode(next);
    } else {
      setPath([...path, currentNode]);
      setCurrentNode(-1);
    }
  };

  const reset = () => {
    setCurrentNode(0);
    setPath([]);
  };

  const goBack = () => {
    if (path.length > 0) {
      const newPath = [...path];
      newPath.pop();
      setPath(newPath);
      setCurrentNode(path[path.length - 1]);
    }
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Tile形状选择指南</h2>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Decision tree */}
        <div className="col-span-2 bg-gray-800 rounded-lg p-6">
          {currentNode === -1 ? (
            <div className="text-center">
              <h3 className="text-2xl font-bold text-green-400 mb-4">推荐配置</h3>
              {Object.entries(recommendations).map(([key, rec]) => (
                <div key={key} className="bg-gray-700 rounded-lg p-4 mb-4 text-left">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-white font-bold">{rec.tile}</span>
                    <span className="text-green-400">{rec.performance}</span>
                  </div>
                  <code className="text-sm text-gray-300 font-mono">{rec.config}</code>
                </div>
              ))}
              <button
                onClick={reset}
                className="mt-4 px-4 py-2 bg-blue-600 rounded-lg text-white hover:bg-blue-500"
              >
                重新选择
              </button>
            </div>
          ) : (
            <div>
              <div className="mb-6">
                <div className="text-gray-400 text-sm mb-2">
                  步骤 {path.length + 1} / {decisions.length}
                </div>
                <h3 className="text-xl text-white font-bold">{decisions[currentNode].question}</h3>
              </div>
              
              <div className="space-y-3">
                {decisions[currentNode].options.map((option, i) => (
                  <button
                    key={i}
                    onClick={() => handleOption(option.next)}
                    className="w-full p-4 bg-gray-700 hover:bg-gray-600 rounded-lg text-left transition-colors"
                  >
                    <span className="text-white">{option.label}</span>
                    <span className="text-gray-400 ml-2">→</span>
                  </button>
                ))}
              </div>
              
              {path.length > 0 && (
                <button
                  onClick={goBack}
                  className="mt-4 px-4 py-2 bg-gray-700 rounded-lg text-gray-300 hover:bg-gray-600"
                >
                  ← 返回上一步
                </button>
              )}
            </div>
          )}
        </div>
        
        {/* Sidebar */}
        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-white font-bold mb-2">选择路径</h3>
            <div className="space-y-2">
              {path.map((nodeIndex, i) => (
                <div key={i} className="text-gray-400 text-sm">
                  {i + 1}. {decisions[nodeIndex].question}
                </div>
              ))}
            </div>
          </div>
          
          <div className="bg-blue-900/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">选择原则</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• 小矩阵用小Tile</li>
              <li>• 大矩阵用大Tile</li>
              <li>• 考虑显存限制</li>
              <li>• Tensor Core需特殊对齐</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

'use client';

import React, { useState } from 'react';

type TilingType = 'row' | 'column' | 'block';

interface TileConfig {
  name: string;
  description: string;
  pattern: number[][];
}

export function TilingStrategyVisualizer() {
  const [strategy, setStrategy] = useState<TilingType>('block');
  const [tileSize, setTileSize] = useState(4);
  const gridSize = 8;

  const strategies: Record<TilingType, TileConfig> = {
    row: {
      name: 'Row Tiling',
      description: '按行分块，每行独立处理',
      pattern: [[1,1,2,2,3,3,4,4],[1,1,2,2,3,3,4,4],[1,1,2,2,3,3,4,4],[1,1,2,2,3,3,4,4],[5,5,6,6,7,7,8,8],[5,5,6,6,7,7,8,8],[5,5,6,6,7,7,8,8],[5,5,6,6,7,7,8,8]],
    },
    column: {
      name: 'Column Tiling',
      description: '按列分块，每列独立处理',
      pattern: [[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[3,3,3,3,4,4,4,4],[3,3,3,3,4,4,4,4],[3,3,3,3,4,4,4,4],[3,3,3,3,4,4,4,4]],
    },
    block: {
      name: 'Block Tiling',
      description: '二维分块，最常用策略',
      pattern: [[1,1,2,2,3,3,4,4],[1,1,2,2,3,3,4,4],[5,5,6,6,7,7,8,8],[5,5,6,6,7,7,8,8],[9,9,10,10,11,11,12,12],[9,9,10,10,11,11,12,12],[13,13,14,14,15,15,16,16],[13,13,14,14,15,15,16,16]],
    },
  };

  const tileColors = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16',
    '#F97316', '#14B8A6', '#A855F7', '#E11D48',
    '#0EA5E9', '#22C55E', '#EAB308', '#DC2626',
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Tiling 策略可视化</h2>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        {(Object.keys(strategies) as TilingType[]).map((type) => (
          <button
            key={type}
            onClick={() => setStrategy(type)}
            className={`p-4 rounded-lg border-2 transition-all ${
              strategy === type
                ? 'border-blue-500 bg-blue-900/30'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="text-white font-bold">{strategies[type].name}</div>
            <div className="text-gray-400 text-sm mt-1">{strategies[type].description}</div>
          </button>
        ))}
      </div>
      
      <div className="flex gap-6">
        {/* Matrix visualization */}
        <div className="flex-1 bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <span className="text-gray-400 text-sm">矩阵分块可视化</span>
            <span className="text-gray-400 text-sm">Tile Size: {tileSize}×{tileSize}</span>
          </div>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
            {strategies[strategy].pattern.flat().map((tileId, i) => (
              <div
                key={i}
                className="w-10 h-10 flex items-center justify-center text-xs font-mono rounded transition-all hover:scale-110"
                style={{
                  backgroundColor: tileColors[(tileId - 1) % tileColors.length],
                }}
              >
                T{tileId}
              </div>
            ))}
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex flex-wrap gap-2">
            {tileColors.slice(0, 4).map((color, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
                <span className="text-gray-400 text-xs">Tile {i + 1}</span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Strategy info */}
        <div className="w-64 space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-white font-bold mb-2">当前策略</h3>
            <div className="text-gray-300 text-sm">
              <p><strong>{strategies[strategy].name}</strong></p>
              <p className="mt-2">{strategies[strategy].description}</p>
            </div>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-white font-bold mb-2">对比</h3>
            <div className="text-gray-300 text-sm space-y-2">
              <div className="flex justify-between">
                <span>Row Tiling</span>
                <span className="text-yellow-400">行局部性</span>
              </div>
              <div className="flex justify-between">
                <span>Column Tiling</span>
                <span className="text-blue-400">列局部性</span>
              </div>
              <div className="flex justify-between">
                <span>Block Tiling</span>
                <span className="text-green-400">二维局部性</span>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-900/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">选择建议</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• GEMM: 使用 Block Tiling</li>
              <li>• 卷积: 使用 Row Tiling</li>
              <li>• 注意力: 使用 Column Tiling</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

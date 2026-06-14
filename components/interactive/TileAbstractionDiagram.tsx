'use client';

import React, { useState } from 'react';

interface Tile {
  id: number;
  x: number;
  y: number;
  color: string;
  thread: number;
}

export function TileAbstractionDiagram() {
  const [selectedTile, setSelectedTile] = useState<number | null>(null);
  const [showThreads, setShowThreads] = useState(true);

  const tileSize = 4;
  const gridSize = 8;
  const tiles: Tile[] = [];

  const tileColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];

  for (let ty = 0; ty < gridSize / tileSize; ty++) {
    for (let tx = 0; tx < gridSize / tileSize; tx++) {
      const tileIndex = ty * (gridSize / tileSize) + tx;
      for (let dy = 0; dy < tileSize; dy++) {
        for (let dx = 0; dx < tileSize; dx++) {
          tiles.push({
            id: tileIndex,
            x: tx * tileSize + dx,
            y: ty * tileSize + dy,
            color: tileColors[tileIndex % tileColors.length],
            thread: tileIndex * tileSize * tileSize + dy * tileSize + dx,
          });
        }
      }
    }
  }

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Tile 抽象示意图</h2>
      
      <div className="flex gap-6">
        {/* Matrix visualization */}
        <div className="flex-1">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <span className="text-gray-400 text-sm">矩阵分块 (Tile Size: {tileSize}×{tileSize})</span>
              <button
                onClick={() => setShowThreads(!showThreads)}
                className="px-3 py-1 bg-gray-700 rounded text-white text-sm hover:bg-gray-600"
              >
                {showThreads ? '隐藏' : '显示'}线程映射
              </button>
            </div>
            
            <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
              {tiles.map((tile) => (
                <div
                  key={`${tile.x}-${tile.y}`}
                  className="w-10 h-10 flex items-center justify-center text-xs font-mono cursor-pointer transition-all hover:scale-110"
                  style={{
                    backgroundColor: tile.color,
                    opacity: selectedTile === null || selectedTile === tile.id ? 1 : 0.3,
                  }}
                  onClick={() => setSelectedTile(selectedTile === tile.id ? null : tile.id)}
                >
                  {showThreads ? tile.thread : ''}
                </div>
              ))}
            </div>
            
            {/* Legend */}
            <div className="mt-4 flex flex-wrap gap-2">
              {tileColors.map((color, i) => (
                <div key={i} className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
                  <span className="text-gray-400 text-xs">Tile {i}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Info panel */}
        <div className="w-64">
          <div className="bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="text-white font-bold mb-2">分块原理</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• 矩阵被划分为多个Tile</li>
              <li>• 每个Tile由一个Block处理</li>
              <li>• Tile大小影响性能</li>
              <li>• 常用大小: 128×128, 64×64</li>
            </ul>
          </div>
          
          {selectedTile !== null && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-white font-bold mb-2">选中Tile {selectedTile}</h3>
              <div className="text-gray-300 text-sm space-y-1">
                <p>• 线程数: {tileSize * tileSize}</p>
                <p>• 元素: {tileSize * tileSize} 个</p>
                <p>• 颜色: {tiles.find(t => t.id === selectedTile)?.color}</p>
              </div>
            </div>
          )}
          
          <div className="mt-4 bg-blue-900/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">关键概念</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• <strong>Tile</strong>: 矩阵的子块</li>
              <li>• <strong>Thread</strong>: 处理单个元素</li>
              <li>• <strong>Block</strong>: 一组线程</li>
              <li>• <strong>Grid</strong>: 所有Block</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

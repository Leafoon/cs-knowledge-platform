'use client';

import React, { useState } from 'react';

interface ThreadMapping {
  threadId: number;
  row: number;
  col: number;
  tileX: number;
  tileY: number;
}

export function ThreadBindingVisualizer() {
  const [hoveredThread, setHoveredThread] = useState<number | null>(null);
  const [blockSize, setBlockSize] = useState(4);

  const gridRows = 8;
  const gridCols = 8;
  const threads: ThreadMapping[] = [];

  for (let ty = 0; ty < gridRows / blockSize; ty++) {
    for (let tx = 0; tx < gridCols / blockSize; tx++) {
      const blockId = ty * (gridCols / blockSize) + tx;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          threads.push({
            threadId: blockId * blockSize * blockSize + dy * blockSize + dx,
            row: ty * blockSize + dy,
            col: tx * blockSize + dx,
            tileX: tx,
            tileY: ty,
          });
        }
      }
    }
  }

  const getThreadColor = (thread: ThreadMapping) => {
    if (hoveredThread === null) return '#4B5563';
    if (thread.threadId === hoveredThread) return '#3B82F6';
    const hovered = threads.find(t => t.threadId === hoveredThread);
    if (hovered && thread.tileX === hovered.tileX && thread.tileY === hovered.tileY) {
      return '#10B981';
    }
    return '#374151';
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">线程绑定可视化器</h2>
      
      <div className="flex gap-6">
        <div className="flex-1">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <span className="text-gray-400 text-sm">2D Grid: {gridRows}×{gridCols} 元素</span>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 text-sm">Block大小:</span>
                <select
                  value={blockSize}
                  onChange={(e) => setBlockSize(Number(e.target.value))}
                  className="bg-gray-700 text-white rounded px-2 py-1 text-sm"
                >
                  <option value={2}>2×2</option>
                  <option value={4}>4×4</option>
                  <option value={8}>8×8</option>
                </select>
              </div>
            </div>
            
            <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}>
              {threads.map((thread) => (
                <div
                  key={thread.threadId}
                  className="w-12 h-12 flex flex-col items-center justify-center rounded text-xs font-mono cursor-pointer transition-all"
                  style={{ backgroundColor: getThreadColor(thread) }}
                  onMouseEnter={() => setHoveredThread(thread.threadId)}
                  onMouseLeave={() => setHoveredThread(null)}
                >
                  <span className="text-white font-bold">{thread.threadId}</span>
                  <span className="text-gray-300 text-[10px]">({thread.row},{thread.col})</span>
                </div>
              ))}
            </div>
            
            {/* Legend */}
            <div className="mt-4 flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-green-500" />
                <span className="text-gray-300">同Block线程</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-blue-500" />
                <span className="text-gray-300">当前线程</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="w-64">
          <div className="bg-gray-800 rounded-lg p-4 mb-4">
            <h3 className="text-white font-bold mb-2">线程映射规则</h3>
            <div className="text-gray-300 text-sm space-y-2">
              <p><strong>Block ID:</strong> (tx, ty)</p>
              <p><strong>Thread ID:</strong> (dx, dy)</p>
              <p><strong>全局位置:</strong> (tx*B+dx, ty*B+dy)</p>
              <p><strong>线程编号:</strong> (ty*B+dy)*N + (tx*B+dx)</p>
            </div>
          </div>
          
          {hoveredThread !== null && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-white font-bold mb-2">线程信息</h3>
              <div className="text-gray-300 text-sm space-y-1">
                <p>• ID: {hoveredThread}</p>
                <p>• 位置: ({threads[hoveredThread]?.row}, {threads[hoveredThread]?.col})</p>
                <p>• Block: ({threads[hoveredThread]?.tileX}, {threads[hoveredThread]?.tileY})</p>
              </div>
            </div>
          )}
          
          <div className="mt-4 bg-blue-900/30 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">CUDA概念</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• <strong>threadIdx</strong>: Block内线程索引</li>
              <li>• <strong>blockIdx</strong>: Grid内Block索引</li>
              <li>• <strong>blockDim</strong>: Block大小</li>
              <li>• <strong>gridDim</strong>: Grid大小</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

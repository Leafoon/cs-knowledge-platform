'use client';

import { useState } from 'react';

const seqLen = 6;
const tokenLabels = ['我', '们', '今', '天', '去', '玩'];

export default function CausalMaskDemo() {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [animated, setAnimated] = useState(false);

  const isAttending = (row: number, col: number) => col <= row;

  const handlePlay = () => {
    setAnimated(true);
    let r = 0;
    const iv = setInterval(() => {
      r++;
      if (r > seqLen) { clearInterval(iv); setTimeout(() => setAnimated(false), 500); }
    }, 300);
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-xl font-bold">因果掩码（Causal Mask）</h2>
          <p className="text-sm text-gray-400">解码时 token 只能关注自身及之前的 token</p>
        </div>
        <button onClick={handlePlay} className="px-3 py-1 bg-blue-600 rounded text-sm hover:bg-blue-500">
          播放生成
        </button>
      </div>

      <div className="flex gap-4 mb-4">
        {/* Token row */}
        <div className="flex gap-1 ml-[52px] mb-1">
          {tokenLabels.map((t, i) => (
            <div key={i} className="w-[60px] text-center text-sm font-bold text-gray-400">{t}</div>
          ))}
        </div>
      </div>

      <div className="flex mb-4">
        {/* Mask matrix */}
        <div>
          {Array.from({ length: seqLen }, (_, row) => (
            <div key={row} className="flex items-center gap-1 mb-1">
              <div className="w-10 text-right text-sm font-bold text-gray-400 pr-2">{tokenLabels[row]}</div>
              <div className="flex gap-1">
                {Array.from({ length: seqLen }, (_, col) => {
                  const attending = isAttending(row, col);
                  const isHovered = hoveredCell?.row === row && hoveredCell?.col === col;
                  return (
                    <div key={col}
                      className="w-[60px] h-10 rounded flex items-center justify-center text-xs font-mono transition-all cursor-pointer"
                      style={{
                        backgroundColor: attending ? '#3B82F630' : '#1F2937',
                        border: isHovered ? '2px solid #3B82F6' : '1px solid',
                        borderColor: attending ? '#3B82F6' : '#374151',
                        transform: isHovered ? 'scale(1.1)' : 'scale(1)',
                      }}
                      onMouseEnter={() => setHoveredCell({ row, col })}
                      onMouseLeave={() => setHoveredCell(null)}>
                      {attending ? (
                        <span className="text-blue-400">✓</span>
                      ) : (
                        <span className="text-gray-600">✗</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {hoveredCell && (
        <div className="bg-gray-800 rounded-lg p-3 text-sm mb-4">
          <span className="text-gray-400">Token "{tokenLabels[hoveredCell.row]}" → </span>
          <span className={isAttending(hoveredCell.row, hoveredCell.col) ? 'text-green-400' : 'text-red-400'}>
            {isAttending(hoveredCell.row, hoveredCell.col)
              ? `可以关注 "${tokenLabels[hoveredCell.col]}"（位置 ${hoveredCell.col} ≤ ${hoveredCell.row}）`
              : `无法关注 "${tokenLabels[hoveredCell.col]}"（位置 ${hoveredCell.col} > ${hoveredCell.row}，未来 token）`}
          </span>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">上三角掩码</div>
          <div className="text-blue-400 font-bold">M[i][j] = j ≤ i</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">自回归生成</div>
          <div className="text-green-400 font-bold">逐 token 解码</div>
        </div>
        <div className="bg-gray-800 rounded p-2 text-center">
          <div className="text-gray-400">掩码位置</div>
          <div className="text-red-400 font-bold">设为 -∞</div>
        </div>
      </div>
    </div>
  );
}

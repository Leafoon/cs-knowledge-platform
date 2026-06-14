'use client';
import { useState } from 'react';

const tileConfigs = [
  { tileW: 64, tileH: 64, blockW: 16, blockH: 16, warpCount: 4, label: '64x64 Tile' },
  { tileW: 32, tileH: 32, blockW: 8, blockH: 8, warpCount: 4, label: '32x32 Tile' },
  { tileW: 128, tileH: 32, blockW: 16, blockH: 8, warpCount: 8, label: '128x32 Tile' },
];

export function ThreadBindingDiagram() {
  const [configIdx, setConfigIdx] = useState(0);
  const cfg = tileConfigs[configIdx];

  const threadsPerBlock = cfg.blockW * cfg.blockH;
  const cellsPerThread = (cfg.tileW * cfg.tileH) / threadsPerBlock;

  const threadGrid = [];
  for (let ty = 0; ty < cfg.blockH; ty++) {
    for (let tx = 0; tx < cfg.blockW; tx++) {
      threadGrid.push({ tx, ty, id: ty * cfg.blockW + tx });
    }
  }

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">线程绑定关系图</h2>

      <div className="flex gap-2 mb-6">
        {tileConfigs.map((c, i) => (
          <button key={i} onClick={() => setConfigIdx(i)}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${
              i === configIdx ? 'bg-cyan-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{c.label}</button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Tile空间 */}
        <div>
          <div className="text-sm text-gray-400 mb-2 font-medium">Tile 空间 ({cfg.tileW}×{cfg.tileH})</div>
          <div className="border border-gray-700 rounded-lg p-3 bg-gray-800">
            <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${Math.min(cfg.tileW / 4, 16)}, 1fr)` }}>
              {Array.from({ length: Math.min(cfg.tileH / 4, 16) }).map((_, row) =>
                Array.from({ length: Math.min(cfg.tileW / 4, 16) }).map((_, col) => {
                  const threadIdx = Math.floor(row / (cfg.blockH / 4)) * Math.ceil(cfg.blockW / (cfg.tileW / Math.min(cfg.tileW / 4, 16))) + Math.floor(col / (cfg.blockW / Math.min(cfg.tileW / 4, 16)));
                  const colors = ['bg-blue-600', 'bg-green-600', 'bg-purple-600', 'bg-orange-600', 'bg-red-600', 'bg-yellow-600', 'bg-cyan-600', 'bg-pink-600'];
                  const colorIdx = (row % Math.ceil(cfg.blockH / 4)) * Math.ceil(cfg.blockW / Math.min(cfg.tileW / 4, 16)) + (col % Math.ceil(cfg.blockW / Math.min(cfg.tileW / 4, 16)));
                  return (
                    <div key={`${row}-${col}`}
                      className={`w-full aspect-square rounded-sm ${colors[colorIdx % colors.length]} opacity-70`}
                      title={`Tile[${row * 4},${col * 4}]`} />
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Block结构 */}
        <div>
          <div className="text-sm text-gray-400 mb-2 font-medium">Block 线程 ({cfg.blockW}×{cfg.blockH}={threadsPerBlock})</div>
          <div className="border border-gray-700 rounded-lg p-3 bg-gray-800">
            <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${cfg.blockW}, 1fr)` }}>
              {threadGrid.map(t => (
                <div key={t.id}
                  className="w-full aspect-square bg-cyan-700 rounded-sm flex items-center justify-center"
                  title={`thread(${t.tx},${t.ty})`}>
                  <span className="text-[8px] text-cyan-200">{t.id}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-500">每线程处理 {cellsPerThread} 个元素</div>
        </div>

        {/* 映射关系 */}
        <div>
          <div className="text-sm text-gray-400 mb-2 font-medium">映射公式</div>
          <div className="border border-gray-700 rounded-lg p-3 bg-gray-800 space-y-2 font-mono text-xs">
            <div className="text-green-400">BlockIdx.x = tileCol</div>
            <div className="text-green-400">BlockIdx.y = tileRow</div>
            <div className="text-blue-400">ThreadIdx.x = localCol</div>
            <div className="text-blue-400">ThreadIdx.y = localRow</div>
            <div className="border-t border-gray-700 pt-2">
              <div className="text-yellow-400">globalRow =</div>
              <div className="text-yellow-300 ml-4">BlockIdx.y × {cfg.blockH} + ThreadIdx.y</div>
            </div>
            <div>
              <div className="text-yellow-400">globalCol =</div>
              <div className="text-yellow-300 ml-4">BlockIdx.x × {cfg.blockW} + ThreadIdx.x</div>
            </div>
            <div className="border-t border-gray-700 pt-2 text-gray-400">
              Warp数量: {cfg.warpCount}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

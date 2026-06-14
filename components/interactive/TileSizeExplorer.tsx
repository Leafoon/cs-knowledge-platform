'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const tileConfigs = [
  { row: 32, col: 32, label: '32×32', desc: '小块，低缓存压力，低并行度' },
  { row: 64, col: 64, label: '64×64', desc: '中等块，平衡缓存与并行' },
  { row: 128, col: 128, label: '128×128', desc: '大块，高缓存压力，高并行度' },
  { row: 256, col: 64, label: '256×64', desc: '非方块，适合长方形矩阵' },
  { row: 64, col: 256, label: '64×256', desc: '非方块，不同访存模式' },
];

const matrixSize = { M: 1024, N: 1024, K: 1024 };

function calcMetrics(tileRow: number, tileCol: number) {
  const { M, N, K } = matrixSize;
  const tilesM = Math.ceil(M / tileRow);
  const tilesN = Math.ceil(N / tileCol);
  const totalTiles = tilesM * tilesN;
  const computePerTile = tileRow * tileCol * K * 2;
  const memPerTile = (tileRow + tileCol) * K * 4;
  const intensity = computePerTile / memPerTile;
  const occupancy = Math.min(1, totalTiles / 2048);
  const cacheL2 = (tileRow * tileCol * 4) / (1024 * 1024);
  return {
    totalTiles,
    computePerTile: (computePerTile / 1e6).toFixed(1),
    memPerTile: (memPerTile / 1024).toFixed(1),
    intensity: intensity.toFixed(1),
    occupancy: (occupancy * 100).toFixed(0),
    cacheL2: cacheL2.toFixed(2),
    tilesM,
    tilesN,
    score: (occupancy * 0.3 + (intensity / 128) * 0.4 + (1 - Math.abs(cacheL2 - 1.5) / 3) * 0.3).toFixed(2),
  };
}

export default function TileSizeExplorer() {
  const [selected, setSelected] = useState(1);
  const [customRow, setCustomRow] = useState(64);
  const [customCol, setCustomCol] = useState(64);
  const [useCustom, setUseCustom] = useState(false);

  const activeTile = useCustom
    ? { row: customRow, col: customCol, label: `${customRow}×${customCol}`, desc: '自定义配置' }
    : tileConfigs[selected];
  const metrics = calcMetrics(activeTile.row, activeTile.col);

  const allMetrics = tileConfigs.map((t) => calcMetrics(t.row, t.col));
  const maxScore = Math.max(...allMetrics.map((m) => parseFloat(m.score)));

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        分块大小（Tile Size）探索器
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        不同 Tile Size 对矩阵乘法性能的影响（矩阵 {matrixSize.M}×{matrixSize.N}×{matrixSize.K}）
      </p>

      <div className="flex flex-wrap gap-2 justify-center mb-4">
        {tileConfigs.map((t, i) => (
          <button
            key={t.label}
            onClick={() => { setSelected(i); setUseCustom(false); }}
            className={`px-3 py-2 rounded-lg text-sm font-mono transition-all ${
              !useCustom && i === selected
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {t.label}
          </button>
        ))}
        <button
          onClick={() => setUseCustom(true)}
          className={`px-3 py-2 rounded-lg text-sm transition-all ${
            useCustom
              ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          自定义
        </button>
      </div>

      {useCustom && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="flex items-center justify-center gap-4 mb-4"
        >
          <label className="text-sm text-gray-400">
            M:
            <input
              type="number"
              value={customRow}
              onChange={(e) => setCustomRow(Number(e.target.value) || 16)}
              className="ml-2 w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm font-mono"
              min={16}
              max={512}
            />
          </label>
          <label className="text-sm text-gray-400">
            N:
            <input
              type="number"
              value={customCol}
              onChange={(e) => setCustomCol(Number(e.target.value) || 16)}
              className="ml-2 w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-sm font-mono"
              min={16}
              max={512}
            />
          </label>
        </motion.div>
      )}

      <div className="bg-gray-800/40 rounded-xl border border-gray-700 p-6 mb-6">
        <div className="text-center mb-3 text-sm text-gray-300">
          分块视图：{activeTile.label} tiles
        </div>
        <div className="flex justify-center">
          <div
            className="grid gap-px bg-gray-700 rounded overflow-hidden"
            style={{
              gridTemplateColumns: `repeat(${Math.min(metrics.tilesN, 16)}, 1fr)`,
              width: '320px',
              height: '320px',
            }}
          >
            {Array.from({ length: Math.min(metrics.tilesN * metrics.tilesM, 256) }).map((_, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.005 }}
                className="bg-gradient-to-br from-indigo-900/60 to-purple-900/60 hover:from-indigo-600/80 hover:to-purple-600/80 transition-colors"
              />
            ))}
          </div>
        </div>
        <div className="text-center mt-2 text-xs text-gray-500">
          {metrics.tilesM}×{metrics.tilesN} = {metrics.totalTiles} 个分块（仅显示前256个）
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          { label: '总分块数', value: metrics.totalTiles, unit: '', color: 'text-blue-400' },
          { label: '计算量/块', value: metrics.computePerTile, unit: 'MFLOP', color: 'text-purple-400' },
          { label: '访存量/块', value: metrics.memPerTile, unit: 'KB', color: 'text-indigo-400' },
          { label: '计算访存比', value: metrics.intensity, unit: 'FLOP/B', color: 'text-emerald-400' },
          { label: '占用率', value: metrics.occupancy, unit: '%', color: 'text-amber-400' },
          { label: 'L2占用', value: metrics.cacheL2, unit: 'MB', color: 'text-cyan-400' },
        ].map((m) => (
          <motion.div
            key={m.label}
            layout
            className="bg-gray-800/60 rounded-lg p-3 border border-gray-700 text-center"
          >
            <div className={`text-lg font-bold font-mono ${m.color}`}>{m.value}</div>
            <div className="text-xs text-gray-500">
              {m.label} {m.unit && <span>({m.unit})</span>}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 bg-gray-800/40 rounded-xl border border-gray-700 p-4">
        <div className="text-sm text-gray-300 mb-3">综合评分对比</div>
        <div className="space-y-2">
          {tileConfigs.map((t, i) => {
            const score = parseFloat(allMetrics[i].score);
            return (
              <div key={t.label} className="flex items-center gap-3">
                <span className="text-xs font-mono text-gray-400 w-16">{t.label}</span>
                <div className="flex-1 bg-gray-900 rounded-full h-4 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(score / maxScore) * 100}%` }}
                    transition={{ delay: i * 0.1, duration: 0.5 }}
                    className={`h-full rounded-full ${
                      i === selected && !useCustom
                        ? 'bg-gradient-to-r from-indigo-500 to-purple-500'
                        : 'bg-gray-600'
                    }`}
                  />
                </div>
                <span className="text-xs font-mono text-gray-500 w-10">{score}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, Plus, Minus } from 'lucide-react'

export default function LoRAMatrixDecomposition() {
  const [rank, setRank] = useState(4)
  const [showDecomposition, setShowDecomposition] = useState(true)

  const d = 8  // 输入维度
  const k = 8  // 输出维度

  const fullParams = d * k
  const loraParams = d * rank + rank * k
  const reduction = ((1 - loraParams / fullParams) * 100).toFixed(1)

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-violet-500" />
          LoRA 矩阵分解可视化
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          将 ΔW 分解为两个低秩矩阵 B 和 A
        </p>
      </div>

      {/* Rank Control */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            秩 (Rank): {rank}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setRank(Math.max(1, rank - 1))}
              disabled={rank <= 1}
              className="p-2 bg-violet-100 hover:bg-violet-200 dark:bg-violet-900/30 dark:hover:bg-violet-900/50 rounded disabled:opacity-50"
            >
              <Minus className="w-4 h-4 text-violet-600 dark:text-violet-400" />
            </button>
            <button
              onClick={() => setRank(Math.min(8, rank + 1))}
              disabled={rank >= 8}
              className="p-2 bg-violet-100 hover:bg-violet-200 dark:bg-violet-900/30 dark:hover:bg-violet-900/50 rounded disabled:opacity-50"
            >
              <Plus className="w-4 h-4 text-violet-600 dark:text-violet-400" />
            </button>
          </div>
        </div>
        <input
          type="range"
          min="1"
          max="8"
          value={rank}
          onChange={(e) => setRank(Number(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Matrix Visualization */}
      <div className="mb-6">
        <div className="flex items-center justify-center gap-8">
          {/* W0 (frozen) */}
          <div className="text-center">
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
              W₀ (冻结)
            </div>
            <div className="inline-block p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border-2 border-slate-300 dark:border-slate-600">
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${k}, 1fr)` }}>
                {Array.from({ length: d * k }).map((_, i) => (
                  <div
                    key={i}
                    className="w-4 h-4 bg-slate-400 dark:bg-slate-600 rounded-sm"
                  />
                ))}
              </div>
            </div>
            <div className="text-xs text-slate-500 mt-2">{d} × {k}</div>
            <div className="text-xs text-slate-500">{fullParams} 参数</div>
          </div>

          {/* Plus Sign */}
          <div className="text-3xl font-bold text-slate-400">+</div>

          {/* LoRA Decomposition */}
          {showDecomposition ? (
            <>
              {/* B matrix */}
              <div className="text-center">
                <div className="text-sm font-semibold text-violet-700 dark:text-violet-300 mb-2">
                  B (可训练)
                </div>
                <div className="inline-block p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg border-2 border-violet-300 dark:border-violet-700">
                  <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${rank}, 1fr)` }}>
                    {Array.from({ length: d * rank }).map((_, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.01 }}
                        className="w-4 h-4 bg-violet-400 dark:bg-violet-600 rounded-sm"
                      />
                    ))}
                  </div>
                </div>
                <div className="text-xs text-violet-600 dark:text-violet-400 mt-2">
                  {d} × {rank}
                </div>
                <div className="text-xs text-violet-600 dark:text-violet-400">
                  {d * rank} 参数
                </div>
              </div>

              {/* Multiply Sign */}
              <div className="text-3xl font-bold text-slate-400">×</div>

              {/* A matrix */}
              <div className="text-center">
                <div className="text-sm font-semibold text-purple-700 dark:text-purple-300 mb-2">
                  A (可训练)
                </div>
                <div className="inline-block p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-2 border-purple-300 dark:border-purple-700">
                  <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${k}, 1fr)` }}>
                    {Array.from({ length: rank * k }).map((_, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.01 }}
                        className="w-4 h-4 bg-purple-400 dark:bg-purple-600 rounded-sm"
                      />
                    ))}
                  </div>
                </div>
                <div className="text-xs text-purple-600 dark:text-purple-400 mt-2">
                  {rank} × {k}
                </div>
                <div className="text-xs text-purple-600 dark:text-purple-400">
                  {rank * k} 参数
                </div>
              </div>
            </>
          ) : (
            <div className="text-center">
              <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                ΔW
              </div>
              <div className="inline-block p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg border-2 border-violet-300 dark:border-violet-700">
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${k}, 1fr)` }}>
                  {Array.from({ length: d * k }).map((_, i) => (
                    <div
                      key={i}
                      className="w-4 h-4 bg-violet-400 dark:bg-violet-600 rounded-sm"
                    />
                  ))}
                </div>
              </div>
              <div className="text-xs text-slate-500 mt-2">{d} × {k}</div>
              <div className="text-xs text-slate-500">{fullParams} 参数</div>
            </div>
          )}
        </div>

        {/* Toggle Button */}
        <div className="text-center mt-4">
          <button
            onClick={() => setShowDecomposition(!showDecomposition)}
            className="px-4 py-2 bg-violet-500 hover:bg-violet-600 text-white rounded-lg transition-colors"
          >
            {showDecomposition ? '隐藏分解' : '显示分解'}
          </button>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">全量微调参数</div>
          <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">
            {fullParams}
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">LoRA 参数</div>
          <div className="text-2xl font-bold text-violet-600 dark:text-violet-400">
            {loraParams}
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">参数减少</div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {reduction}%
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-sm text-green-400 font-mono text-center">
          h = W₀x + ΔWx = W₀x + BAx
        </div>
        <div className="text-xs text-slate-400 text-center mt-2">
          其中 B ∈ ℝ<sup>{d}×{rank}</sup>, A ∈ ℝ<sup>{rank}×{k}</sup>
        </div>
      </div>
    </div>
  )
}

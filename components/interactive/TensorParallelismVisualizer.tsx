'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Grid3X3, ArrowRight, Plus } from 'lucide-react'

type ParallelMode = 'column' | 'row'

export default function TensorParallelismVisualizer() {
  const [mode, setMode] = useState<ParallelMode>('column')

  // 矩阵尺寸
  const batchSize = 4
  const hiddenSize = 8
  const ffnSize = 16

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-pink-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Grid3X3 className="w-8 h-8 text-pink-600" />
        <h3 className="text-2xl font-bold text-slate-800">张量并行可视化</h3>
      </div>

      {/* 模式选择 */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => setMode('column')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'column'
              ? 'border-pink-600 bg-pink-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-pink-300'
          }`}
        >
          <div className={`font-bold mb-2 ${
            mode === 'column' ? 'text-pink-900' : 'text-slate-700'
          }`}>
            列并行 (Column Parallel)
          </div>
          <div className="text-sm text-slate-600">
            权重矩阵沿列切分，输出拼接
          </div>
        </button>

        <button
          onClick={() => setMode('row')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'row'
              ? 'border-pink-600 bg-pink-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-pink-300'
          }`}
        >
          <div className={`font-bold mb-2 ${
            mode === 'row' ? 'text-pink-900' : 'text-slate-700'
          }`}>
            行并行 (Row Parallel)
          </div>
          <div className="text-sm text-slate-600">
            权重矩阵沿行切分，输出求和
          </div>
        </button>
      </div>

      {/* 列并行可视化 */}
      {mode === 'column' && (
        <motion.div
          key="column"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* 公式 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="font-mono text-center text-slate-700 mb-4">
              Y = X @ W  →  Y = concat([X @ W₁, X @ W₂])
            </div>
            <div className="text-sm text-center text-slate-600">
              W ∈ ℝ<sup>{hiddenSize}×{ffnSize}</sup> 切分为 W₁, W₂ ∈ ℝ<sup>{hiddenSize}×{ffnSize/2}</sup>
            </div>
          </div>

          {/* 矩阵运算可视化 */}
          <div className="flex items-center justify-center gap-6">
            {/* 输入 X */}
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-slate-700 mb-2">X</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: batchSize * hiddenSize }).map((_, i) => (
                  <div
                    key={i}
                    className="w-6 h-6 bg-gradient-to-br from-blue-400 to-blue-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {hiddenSize}]</div>
            </div>

            <div className="text-2xl text-slate-400">@</div>

            {/* GPU 0: W₁ */}
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-pink-700 mb-2">W₁ (GPU 0)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: hiddenSize * ffnSize / 2 }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.005 }}
                    className="w-4 h-4 bg-gradient-to-br from-pink-400 to-pink-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{hiddenSize}, {ffnSize/2}]</div>
            </div>

            <div className="text-2xl text-slate-400">=</div>

            {/* 输出 Y₁ */}
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-green-700 mb-2">Y₁</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: batchSize * ffnSize / 2 }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + i * 0.01 }}
                    className="w-6 h-6 bg-gradient-to-br from-green-400 to-green-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {ffnSize/2}]</div>
            </div>
          </div>

          {/* GPU 1 计算 */}
          <div className="flex items-center justify-center gap-6">
            <div className="w-20" /> {/* 占位 */}
            <div className="text-2xl text-slate-400">@</div>

            {/* GPU 1: W₂ */}
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-purple-700 mb-2">W₂ (GPU 1)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: hiddenSize * ffnSize / 2 }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.005 }}
                    className="w-4 h-4 bg-gradient-to-br from-purple-400 to-purple-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{hiddenSize}, {ffnSize/2}]</div>
            </div>

            <div className="text-2xl text-slate-400">=</div>

            {/* 输出 Y₂ */}
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-orange-700 mb-2">Y₂</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: batchSize * ffnSize / 2 }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + i * 0.01 }}
                    className="w-6 h-6 bg-gradient-to-br from-orange-400 to-orange-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {ffnSize/2}]</div>
            </div>
          </div>

          {/* 拼接 */}
          <div className="flex items-center justify-center gap-4">
            <div className="text-lg font-bold text-slate-700">Concat</div>
            <ArrowRight className="w-6 h-6 text-slate-400" />
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-blue-700 mb-2">Y (最终输出)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize}, 1fr)` }}>
                {Array.from({ length: batchSize * ffnSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.6 + i * 0.005 }}
                    className={`w-4 h-4 rounded-sm ${
                      i % ffnSize < ffnSize / 2
                        ? 'bg-gradient-to-br from-green-400 to-green-600'
                        : 'bg-gradient-to-br from-orange-400 to-orange-600'
                    }`}
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {ffnSize}]</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 行并行可视化 */}
      {mode === 'row' && (
        <motion.div
          key="row"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* 公式 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="font-mono text-center text-slate-700 mb-4">
              Y = X @ W  →  Y = (X₁ @ W₁) + (X₂ @ W₂)
            </div>
            <div className="text-sm text-center text-slate-600">
              W ∈ ℝ<sup>{ffnSize}×{hiddenSize}</sup> 切分为 W₁, W₂ ∈ ℝ<sup>{ffnSize/2}×{hiddenSize}</sup>
            </div>
          </div>

          {/* GPU 0 计算 */}
          <div className="flex items-center justify-center gap-6">
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-blue-700 mb-2">X₁ (GPU 0)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: batchSize * ffnSize / 2 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-6 h-6 bg-gradient-to-br from-blue-400 to-blue-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {ffnSize/2}]</div>
            </div>

            <div className="text-2xl text-slate-400">@</div>

            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-pink-700 mb-2">W₁</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: (ffnSize/2) * hiddenSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.003 }}
                    className="w-4 h-4 bg-gradient-to-br from-pink-400 to-pink-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{ffnSize/2}, {hiddenSize}]</div>
            </div>

            <div className="text-2xl text-slate-400">=</div>

            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-green-700 mb-2">Y₁</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: batchSize * hiddenSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + i * 0.01 }}
                    className="w-6 h-6 bg-gradient-to-br from-green-400 to-green-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {hiddenSize}]</div>
            </div>
          </div>

          {/* GPU 1 计算 */}
          <div className="flex items-center justify-center gap-6">
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-purple-700 mb-2">X₂ (GPU 1)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${ffnSize/2}, 1fr)` }}>
                {Array.from({ length: batchSize * ffnSize / 2 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-6 h-6 bg-gradient-to-br from-purple-400 to-purple-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {ffnSize/2}]</div>
            </div>

            <div className="text-2xl text-slate-400">@</div>

            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-orange-700 mb-2">W₂</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: (ffnSize/2) * hiddenSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.003 }}
                    className="w-4 h-4 bg-gradient-to-br from-orange-400 to-orange-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{ffnSize/2}, {hiddenSize}]</div>
            </div>

            <div className="text-2xl text-slate-400">=</div>

            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-cyan-700 mb-2">Y₂</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: batchSize * hiddenSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + i * 0.01 }}
                    className="w-6 h-6 bg-gradient-to-br from-cyan-400 to-cyan-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {hiddenSize}]</div>
            </div>
          </div>

          {/* AllReduce求和 */}
          <div className="flex items-center justify-center gap-4">
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium text-green-700">Y₁</div>
              <Plus className="w-5 h-5 text-slate-400" />
              <div className="text-sm font-medium text-cyan-700">Y₂</div>
            </div>
            <div className="text-lg font-bold text-slate-700">(AllReduce)</div>
            <ArrowRight className="w-6 h-6 text-slate-400" />
            <div className="flex flex-col items-center">
              <div className="text-sm font-medium text-indigo-700 mb-2">Y (最终输出)</div>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${hiddenSize}, 1fr)` }}>
                {Array.from({ length: batchSize * hiddenSize }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.6 + i * 0.01 }}
                    className="w-6 h-6 bg-gradient-to-br from-indigo-400 to-indigo-600 rounded-sm"
                  />
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-2">[{batchSize}, {hiddenSize}]</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 对比表格 */}
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">列并行 vs 行并行</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300">
              <th className="text-left py-2 px-4">特性</th>
              <th className="text-left py-2 px-4">列并行</th>
              <th className="text-left py-2 px-4">行并行</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-200">
              <td className="py-2 px-4 font-medium">权重切分</td>
              <td className="py-2 px-4">沿列切分 (输出维度)</td>
              <td className="py-2 px-4">沿行切分 (输入维度)</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-2 px-4 font-medium">输入数据</td>
              <td className="py-2 px-4">所有GPU相同</td>
              <td className="py-2 px-4">需要切分</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-2 px-4 font-medium">输出合并</td>
              <td className="py-2 px-4">Concat 拼接</td>
              <td className="py-2 px-4">AllReduce 求和</td>
            </tr>
            <tr>
              <td className="py-2 px-4 font-medium">典型应用</td>
              <td className="py-2 px-4">FFN第一层, Q/K/V投影</td>
              <td className="py-2 px-4">FFN第二层, 输出投影</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

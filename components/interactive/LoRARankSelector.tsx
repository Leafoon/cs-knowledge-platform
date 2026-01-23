'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Sliders, TrendingUp } from 'lucide-react'

export default function LoRARankSelector() {
  const [rank, setRank] = useState(8)
  const [alpha, setAlpha] = useState(16)

  // 模拟性能数据
  const getMetrics = (r: number) => {
    const baseAccuracy = 93.5
    const accuracyLoss = Math.max(0, (8 - r) * 0.15)
    const accuracy = baseAccuracy - accuracyLoss

    const params = (768 * r + r * 768) / 1000
    const memory = 4.5 + (r * 0.15)
    const speed = 100 - (r * 2)

    return { accuracy, params, memory, speed }
  }

  const metrics = getMetrics(rank)
  const scalingFactor = alpha / rank

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Sliders className="w-5 h-5 text-cyan-500" />
          LoRA 超参数选择器
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          交互式调整 rank 和 alpha 参数
        </p>
      </div>

      {/* Controls */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Rank Slider */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              Rank (r)
            </span>
            <span className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
              {rank}
            </span>
          </div>
          <input
            type="range"
            min="2"
            max="64"
            step="2"
            value={rank}
            onChange={(e) => setRank(Number(e.target.value))}
            className="w-full mb-2"
          />
          <div className="flex justify-between text-xs text-slate-500">
            <span>2</span>
            <span>32</span>
            <span>64</span>
          </div>
          <div className="mt-3 text-xs text-slate-600 dark:text-slate-400">
            矩阵分解的秩，控制表达能力
          </div>
        </div>

        {/* Alpha Slider */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              LoRA Alpha (α)
            </span>
            <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {alpha}
            </span>
          </div>
          <input
            type="range"
            min="4"
            max="128"
            step="4"
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-full mb-2"
          />
          <div className="flex justify-between text-xs text-slate-500">
            <span>4</span>
            <span>64</span>
            <span>128</span>
          </div>
          <div className="mt-3 text-xs text-slate-600 dark:text-slate-400">
            缩放因子，推荐设为 r 的 2 倍
          </div>
        </div>
      </div>

      {/* Scaling Factor */}
      <div className="mb-6 p-4 bg-gradient-to-r from-cyan-100 to-blue-100 dark:from-cyan-900/30 dark:to-blue-900/30 rounded-lg border border-cyan-200 dark:border-cyan-800">
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-2">
          实际缩放系数: <span className="font-mono font-bold">α / r = {alpha} / {rank} = {scalingFactor.toFixed(2)}</span>
        </div>
        <div className="text-xs text-slate-600 dark:text-slate-400">
          推荐值: 2.0（α = 2 × r），当前: {scalingFactor.toFixed(2)}
          {scalingFactor === 2 && ' ✓ 完美!'}
          {scalingFactor < 1.5 && ' ⚠️ 可能过小'}
          {scalingFactor > 3 && ' ⚠️ 可能过大'}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <motion.div
          key={`acc-${rank}`}
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400">准确率</span>
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {metrics.accuracy.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            (全量: 93.5%)
          </div>
        </motion.div>

        <motion.div
          key={`params-${rank}`}
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-2">参数量</div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {metrics.params.toFixed(1)}K
          </div>
          <div className="text-xs text-slate-500 mt-1">
            (全量: 589K)
          </div>
        </motion.div>

        <motion.div
          key={`memory-${rank}`}
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-2">显存占用</div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {metrics.memory.toFixed(1)} GB
          </div>
          <div className="text-xs text-slate-500 mt-1">
            (全量: 8.5 GB)
          </div>
        </motion.div>

        <motion.div
          key={`speed-${rank}`}
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-2">训练速度</div>
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {metrics.speed}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            (相对全量微调)
          </div>
        </motion.div>
      </div>

      {/* Recommendations */}
      <div className="p-4 bg-slate-900 rounded-lg">
        <div className="text-sm font-semibold text-green-400 mb-2">💡 推荐配置</div>
        <div className="space-y-1 text-xs text-slate-300 font-mono">
          <div>小模型 (&lt;1B)    → r=4~8,   alpha=8~16</div>
          <div>中等模型 (1-7B)   → r=8~16,  alpha=16~32  ← 当前选择</div>
          <div>大模型 (&gt;7B)     → r=16~32, alpha=32~64</div>
        </div>
      </div>
    </div>
  )
}

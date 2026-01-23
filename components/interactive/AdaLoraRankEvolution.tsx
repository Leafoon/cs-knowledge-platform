'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingDown, Clock } from 'lucide-react'

interface RankData {
  epoch: number
  layers: number[]
}

export default function AdaLoraRankEvolution() {
  const [currentEpoch, setCurrentEpoch] = useState(0)

  // 模拟12层的秩演化（从epoch 0到10）
  const rankEvolution: RankData[] = [
    { epoch: 0, layers: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12] },
    { epoch: 2, layers: [11, 11, 12, 12, 12, 12, 12, 12, 12, 11, 10, 10] },
    { epoch: 4, layers: [10, 10, 11, 12, 12, 12, 12, 12, 11, 10, 8, 8] },
    { epoch: 6, layers: [8, 9, 10, 12, 12, 12, 12, 12, 10, 9, 6, 6] },
    { epoch: 8, layers: [6, 8, 9, 11, 12, 12, 12, 11, 9, 8, 4, 4] },
    { epoch: 10, layers: [4, 6, 8, 10, 12, 12, 12, 10, 8, 6, 4, 4] }
  ]

  const currentData = rankEvolution[currentEpoch]
  const avgRank = currentData.layers.reduce((a, b) => a + b, 0) / currentData.layers.length
  const totalParams = currentData.layers.reduce((sum, r) => sum + (768 * r + r * 768), 0)

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <TrendingDown className="w-5 h-5 text-teal-500" />
          AdaLoRA 动态秩演化
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          根据重要性自动调整不同层的秩
        </p>
      </div>

      {/* Epoch Slider */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-teal-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              训练进度: Epoch {currentData.epoch}
            </span>
          </div>
          <span className="text-xs text-slate-500">
            平均秩: {avgRank.toFixed(1)}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max={rankEvolution.length - 1}
          value={currentEpoch}
          onChange={(e) => setCurrentEpoch(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>开始</span>
          <span>中期</span>
          <span>结束</span>
        </div>
      </div>

      {/* Layer Ranks Visualization */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-4">
          各层秩分布
        </div>
        <div className="space-y-2">
          {currentData.layers.map((rank, idx) => {
            const importance = rank / 12
            const color = importance > 0.8
              ? 'bg-red-500'
              : importance > 0.6
              ? 'bg-orange-500'
              : importance > 0.4
              ? 'bg-yellow-500'
              : importance > 0.2
              ? 'bg-teal-500'
              : 'bg-blue-500'

            return (
              <div key={idx} className="flex items-center gap-3">
                <span className="text-xs text-slate-600 dark:text-slate-400 w-16">
                  Layer {idx}
                </span>
                <div className="flex-1">
                  <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(rank / 12) * 100}%` }}
                      transition={{ duration: 0.3 }}
                      className={`h-full ${color} flex items-center justify-end pr-2`}
                    >
                      <span className="text-xs font-bold text-white">
                        r={rank}
                      </span>
                    </motion.div>
                  </div>
                </div>
                <span className="text-xs text-slate-500 w-12 text-right">
                  {((rank / 12) * 100).toFixed(0)}%
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">平均秩</div>
          <div className="text-2xl font-bold text-teal-600 dark:text-teal-400">
            {avgRank.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500 mt-1">初始: 12.0</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">总参数量</div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {(totalParams / 1000).toFixed(1)}K
          </div>
          <div className="text-xs text-slate-500 mt-1">初始: 147K</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">参数减少</div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {((1 - totalParams / 147456) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">相比初始</div>
        </div>
      </div>

      {/* Insights */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="text-sm font-semibold text-red-700 dark:text-red-300 mb-2">
            🔥 重要层（高秩）
          </div>
          <div className="text-xs text-red-600 dark:text-red-400">
            中间层（Layer 4-7）保持高秩，因为它们捕获最关键的特征转换
          </div>
        </div>
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
            ❄️ 次要层（低秩）
          </div>
          <div className="text-xs text-blue-600 dark:text-blue-400">
            首尾层（Layer 0-1, 10-11）秩降低，减少不必要的参数
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">AdaLoRA 配置</div>
        <div className="font-mono text-sm text-green-400">
          <div>AdaLoraConfig(</div>
          <div className="ml-4">r=8,  # 初始秩</div>
          <div className="ml-4">target_r=4,  # 目标平均秩</div>
          <div className="ml-4">init_r=12,  # 初始化秩</div>
          <div className="ml-4">tinit=200,  # 预热步数</div>
          <div className="ml-4">tfinal=1000,  # 最终步数</div>
          <div className="ml-4">deltaT=10  # 调整间隔</div>
          <div>)</div>
        </div>
      </div>
    </div>
  )
}

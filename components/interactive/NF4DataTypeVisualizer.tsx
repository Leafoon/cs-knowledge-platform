'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Database } from 'lucide-react'

export default function NF4DataTypeVisualizer() {
  // NF4 的 16 个量化级别
  const nf4Levels = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
  ]

  // INT4 均匀分布级别
  const int4Levels = Array.from({ length: 16 }, (_, i) => -1 + (i / 15) * 2)

  const maxHeight = 120

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-indigo-500" />
          NF4 vs INT4 量化级别对比
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          NF4 针对正态分布优化，INT4 均匀分布
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* NF4 Distribution */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 bg-indigo-500 rounded"></div>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              NF4 (Normal Float 4-bit)
            </span>
          </div>
          <div className="flex items-end justify-between h-32 gap-0.5">
            {nf4Levels.map((level, idx) => {
              const height = ((level + 1) / 2) * maxHeight
              return (
                <motion.div
                  key={idx}
                  initial={{ height: 0 }}
                  animate={{ height: `${height}px` }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex-1 bg-indigo-500 rounded-t hover:bg-indigo-600 transition-colors relative group"
                  title={`Level ${idx}: ${level.toFixed(3)}`}
                >
                  <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-slate-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {level.toFixed(3)}
                  </div>
                </motion.div>
              )
            })}
          </div>
          <div className="text-xs text-center text-slate-500 mt-2">
            密集分布在 [-0.3, 0.3] 区间（权重集中区域）
          </div>
        </div>

        {/* INT4 Distribution */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 bg-slate-500 rounded"></div>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              INT4 (传统均匀量化)
            </span>
          </div>
          <div className="flex items-end justify-between h-32 gap-0.5">
            {int4Levels.map((level, idx) => {
              const height = ((level + 1) / 2) * maxHeight
              return (
                <motion.div
                  key={idx}
                  initial={{ height: 0 }}
                  animate={{ height: `${height}px` }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex-1 bg-slate-500 rounded-t hover:bg-slate-600 transition-colors relative group"
                  title={`Level ${idx}: ${level.toFixed(3)}`}
                >
                  <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-slate-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {level.toFixed(3)}
                  </div>
                </motion.div>
              )
            })}
          </div>
          <div className="text-xs text-center text-slate-500 mt-2">
            均匀分布在 [-1, 1] 区间
          </div>
        </div>
      </div>

      {/* Comparison Stats */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">
            <BarChart3 className="w-4 h-4 inline mr-1" />
            量化级别
          </div>
          <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">
            16
          </div>
          <div className="text-xs text-slate-500 mt-1">两种方法相同</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">量化误差 (MSE)</div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            -27%
          </div>
          <div className="text-xs text-slate-500 mt-1">NF4 比 INT4 更低</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">存储大小</div>
          <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
            4 bit
          </div>
          <div className="text-xs text-slate-500 mt-1">每个参数</div>
        </div>
      </div>

      {/* Explanation */}
      <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg">
        <div className="text-sm font-semibold text-indigo-700 dark:text-indigo-300 mb-2">
          💡 为什么 NF4 更适合神经网络？
        </div>
        <div className="text-sm text-indigo-600 dark:text-indigo-400 space-y-1">
          <div>• 神经网络权重通常服从正态分布（均值 0，标准差 0.02-0.1）</div>
          <div>• NF4 在 [-0.3, 0.3] 区间有更多量化级别，精度更高</div>
          <div>• INT4 在极值区域浪费了量化级别</div>
          <div>• 实验表明 NF4 量化误差比 INT4 低约 27%</div>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">QLoRA 中的 NF4 配置</div>
        <div className="font-mono text-sm text-green-400">
          <div>BitsAndBytesConfig(</div>
          <div className="ml-4">load_in_4bit=True,</div>
          <div className="ml-4">bnb_4bit_quant_type=<span className="text-yellow-400">&quot;nf4&quot;</span>,  # ← 使用 NF4</div>
          <div className="ml-4">bnb_4bit_compute_dtype=torch.bfloat16</div>
          <div>)</div>
        </div>
      </div>
    </div>
  )
}

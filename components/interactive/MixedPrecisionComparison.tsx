'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, MemoryStick, Gauge, TrendingUp } from 'lucide-react'

interface PrecisionMode {
  id: string
  name: string
  bits: number
  memory: number
  speed: number
  stability: number
  color: string
  description: string
  requirements: string
}

export default function MixedPrecisionComparison() {
  const [selected, setSelected] = useState('fp32')

  const modes: PrecisionMode[] = [
    {
      id: 'fp32',
      name: 'FP32',
      bits: 32,
      memory: 100,
      speed: 100,
      stability: 100,
      color: 'slate',
      description: '单精度浮点数（默认）',
      requirements: '无特殊要求，所有硬件支持'
    },
    {
      id: 'fp16',
      name: 'FP16',
      bits: 16,
      memory: 50,
      speed: 250,
      stability: 70,
      color: 'blue',
      description: '半精度浮点数（需 loss scaling）',
      requirements: 'Nvidia GPU（Volta+ 架构）'
    },
    {
      id: 'bf16',
      name: 'BF16',
      bits: 16,
      memory: 50,
      speed: 250,
      stability: 95,
      color: 'green',
      description: 'Brain Float 16（推荐）',
      requirements: 'Nvidia Ampere+（A100/RTX 30/40 系）'
    }
  ]

  const selectedMode = modes.find(m => m.id === selected)!

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-cyan-50 to-sky-50 dark:from-slate-900 dark:to-cyan-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-cyan-500" />
          混合精度训练对比
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          FP32 vs FP16 vs BF16 性能与显存分析
        </p>
      </div>

      {/* Mode Selection */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {modes.map(mode => (
          <button
            key={mode.id}
            onClick={() => setSelected(mode.id)}
            className={`p-4 rounded-lg border-2 transition-all text-left ${
              selected === mode.id
                ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-900/20'
                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800'
            }`}
          >
            <div className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
              {mode.name}
            </div>
            <div className="text-xs text-slate-600 dark:text-slate-400">
              {mode.bits} bits
            </div>
          </button>
        ))}
      </div>

      {/* Detailed Comparison */}
      <motion.div
        key={selected}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="space-y-6"
      >
        {/* Description */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="font-semibold text-slate-900 dark:text-white mb-2">
            {selectedMode.description}
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">
            <span className="font-medium">硬件要求:</span> {selectedMode.requirements}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid md:grid-cols-3 gap-4">
          {/* Memory */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <MemoryStick className="w-5 h-5 text-blue-500" />
              <span className="font-semibold text-slate-900 dark:text-white">显存占用</span>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-blue-500">{selectedMode.memory}%</span>
                <span className="text-xs text-slate-500">vs FP32</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${selectedMode.memory}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-2 bg-blue-500 rounded-full"
                />
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">
                {selected === 'fp32' && '基准: ~14GB (BERT-Base)'}
                {selected === 'fp16' && '节省: ~7GB'}
                {selected === 'bf16' && '节省: ~7GB'}
              </div>
            </div>
          </div>

          {/* Speed */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Gauge className="w-5 h-5 text-green-500" />
              <span className="font-semibold text-slate-900 dark:text-white">训练速度</span>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-green-500">{selectedMode.speed}%</span>
                <span className="text-xs text-slate-500">vs FP32</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, selectedMode.speed / 2.5)}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-2 bg-green-500 rounded-full"
                />
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">
                {selected === 'fp32' && '基准: 100 samples/s'}
                {selected === 'fp16' && '加速: 2-3x (250 samples/s)'}
                {selected === 'bf16' && '加速: 2-3x (250 samples/s)'}
              </div>
            </div>
          </div>

          {/* Stability */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-5 h-5 text-purple-500" />
              <span className="font-semibold text-slate-900 dark:text-white">数值稳定性</span>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-purple-500">{selectedMode.stability}%</span>
                <span className="text-xs text-slate-500">稳定性</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${selectedMode.stability}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-2 bg-purple-500 rounded-full"
                />
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">
                {selected === 'fp32' && '最佳: 无精度损失'}
                {selected === 'fp16' && '可能溢出: 需 loss scaling'}
                {selected === 'bf16' && '优秀: 更大动态范围'}
              </div>
            </div>
          </div>
        </div>

        {/* Code Example */}
        <div className="p-4 bg-slate-900 rounded-lg">
          <div className="text-xs text-slate-400 mb-2">TrainingArguments 配置</div>
          <pre className="text-sm text-slate-100 font-mono overflow-auto">
            <code>{`training_args = TrainingArguments(
    output_dir="./results",
    ${selected === 'fp16' ? 'fp16=True,  # 启用 FP16 混合精度' : ''}
    ${selected === 'bf16' ? 'bf16=True,  # 启用 BF16 混合精度（推荐）' : ''}
    ${selected === 'fp32' ? '# fp16=False, bf16=False  # 使用 FP32（默认）' : ''}
)`}</code>
          </pre>
        </div>
      </motion.div>
    </div>
  )
}

'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Zap, HardDrive } from 'lucide-react'

export default function MixedPrecisionBenchmark() {
  const [selectedModel, setSelectedModel] = useState('LLaMA-7B')

  const benchmarks = {
    'BERT-Base': {
      params: '110M',
      fp32: { speed: 120, memory: 8, accuracy: 88.2 },
      bf16: { speed: 280, memory: 4.5, accuracy: 88.2 },
      speedup: 2.33,
      memorySave: 44,
      note: undefined,
    },
    'BERT-Large': {
      params: '340M',
      fp32: { speed: 45, memory: 18, accuracy: 87.5 },
      bf16: { speed: 110, memory: 9, accuracy: 87.4 },
      speedup: 2.44,
      memorySave: 50,
      note: undefined,
    },
    'GPT-2': {
      params: '1.5B',
      fp32: { speed: 18, memory: 32, accuracy: 0 },
      bf16: { speed: 42, memory: 16, accuracy: 0 },
      speedup: 2.33,
      memorySave: 50,
      note: '仅预训练，无微调准确率',
    },
    'LLaMA-7B': {
      params: '7B',
      fp32: { speed: 8, memory: 70, accuracy: 0 },
      bf16: { speed: 22, memory: 45, accuracy: 0 },
      speedup: 2.75,
      memorySave: 36,
      note: 'FP32 在单卡 A100 80GB 可训练',
    },
    'LLaMA-13B': {
      params: '13B',
      fp32: { speed: 0, memory: 0, accuracy: 0 },
      bf16: { speed: 12, memory: 78, accuracy: 0 },
      speedup: 0,
      memorySave: 0,
      note: 'FP32 OOM，仅 BF16 可训练',
    },
  }

  const current = benchmarks[selectedModel as keyof typeof benchmarks]
  const canTrainFP32 = current.fp32.speed > 0

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <BarChart3 className="w-8 h-8 text-green-600" />
        <h3 className="text-2xl font-bold text-slate-800">混合精度性能基准测试</h3>
      </div>

      {/* 模型选择 */}
      <div className="grid grid-cols-5 gap-2 mb-6">
        {Object.keys(benchmarks).map((model) => (
          <button
            key={model}
            onClick={() => setSelectedModel(model)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedModel === model
                ? 'border-green-600 bg-green-50 shadow-lg'
                : 'border-slate-200 bg-white hover:border-green-300'
            }`}
          >
            <div className={`font-bold text-sm ${
              selectedModel === model ? 'text-green-900' : 'text-slate-700'
            }`}>
              {model}
            </div>
            <div className="text-xs text-slate-600 mt-1">
              {benchmarks[model as keyof typeof benchmarks].params}
            </div>
          </button>
        ))}
      </div>

      {/* 性能对比 */}
      <motion.div
        key={selectedModel}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* 训练速度对比 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-6 h-6 text-yellow-600" />
            <h4 className="font-bold text-slate-800">训练速度（samples/s）</h4>
          </div>

          <div className="space-y-4">
            {/* FP32 */}
            {canTrainFP32 ? (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-slate-700">FP32 (Float32)</span>
                  <span className="font-mono text-lg font-bold text-blue-600">
                    {current.fp32.speed} samples/s
                  </span>
                </div>
                <div className="h-10 bg-slate-100 rounded-lg overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(current.fp32.speed / 300) * 100}%` }}
                    className="h-full bg-blue-500 flex items-center justify-center"
                  >
                    <span className="text-white text-sm font-bold">基准</span>
                  </motion.div>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-red-50 border border-red-300 rounded-lg">
                <span className="text-red-700 font-semibold">
                  ⚠️ FP32 显存不足（OOM）
                </span>
              </div>
            )}

            {/* BF16 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-700">BF16 (BFloat16)</span>
                <span className="font-mono text-lg font-bold text-green-600">
                  {current.bf16.speed} samples/s
                  {canTrainFP32 && (
                    <span className="text-sm text-green-600 ml-2">
                      ({current.speedup.toFixed(2)}x faster)
                    </span>
                  )}
                </span>
              </div>
              <div className="h-10 bg-slate-100 rounded-lg overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(current.bf16.speed / 300) * 100}%` }}
                  className="h-full bg-green-500 flex items-center justify-center"
                >
                  <Zap className="w-5 h-5 text-white" />
                </motion.div>
              </div>
            </div>
          </div>
        </div>

        {/* 显存占用对比 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <HardDrive className="w-6 h-6 text-purple-600" />
            <h4 className="font-bold text-slate-800">显存占用（单卡 A100 80GB）</h4>
          </div>

          <div className="space-y-4">
            {/* FP32 */}
            {canTrainFP32 ? (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-slate-700">FP32</span>
                  <span className="font-mono text-lg font-bold text-blue-600">
                    {current.fp32.memory} GB
                  </span>
                </div>
                <div className="h-10 bg-slate-100 rounded-lg overflow-hidden relative">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(current.fp32.memory / 80) * 100}%` }}
                    className={`h-full ${
                      current.fp32.memory > 80 ? 'bg-red-500' : 'bg-blue-500'
                    } flex items-center justify-center`}
                  >
                    <span className="text-white text-sm font-bold">
                      {((current.fp32.memory / 80) * 100).toFixed(0)}%
                    </span>
                  </motion.div>
                  <div className="absolute right-0 top-0 h-full w-0.5 bg-red-500" style={{ left: '100%' }}>
                    <span className="absolute -top-6 -right-8 text-xs text-red-600">80GB</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-red-50 border border-red-300 rounded-lg">
                <span className="text-red-700 font-semibold">
                  ⚠️ 超过 80GB 显存限制（需多卡或量化）
                </span>
              </div>
            )}

            {/* BF16 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-700">BF16</span>
                <span className="font-mono text-lg font-bold text-green-600">
                  {current.bf16.memory} GB
                  {canTrainFP32 && (
                    <span className="text-sm text-green-600 ml-2">
                      (节省 {current.memorySave}%)
                    </span>
                  )}
                </span>
              </div>
              <div className="h-10 bg-slate-100 rounded-lg overflow-hidden relative">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(current.bf16.memory / 80) * 100}%` }}
                  className="h-full bg-green-500 flex items-center justify-center"
                >
                  <span className="text-white text-sm font-bold">
                    {((current.bf16.memory / 80) * 100).toFixed(0)}%
                  </span>
                </motion.div>
              </div>
            </div>
          </div>
        </div>

        {/* 准确率对比（如果有）*/}
        {current.fp32.accuracy > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h4 className="font-bold text-slate-800 mb-4">GLUE 准确率（MNLI）</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded border border-blue-200">
                <div className="text-sm text-slate-600 mb-1">FP32</div>
                <div className="text-3xl font-bold text-blue-600">
                  {current.fp32.accuracy}%
                </div>
              </div>
              <div className="p-4 bg-green-50 rounded border border-green-200">
                <div className="text-sm text-slate-600 mb-1">BF16</div>
                <div className="text-3xl font-bold text-green-600">
                  {current.bf16.accuracy}%
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  差异: {Math.abs(current.fp32.accuracy - current.bf16.accuracy).toFixed(1)}%
                  {Math.abs(current.fp32.accuracy - current.bf16.accuracy) < 0.2 && (
                    <span className="text-green-600 ml-1">（几乎无损）</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 备注 */}
        {current.note && (
          <div className="p-4 bg-yellow-50 border border-yellow-300 rounded-lg">
            <div className="text-sm text-slate-700">
              📝 <strong>说明</strong>: {current.note}
            </div>
          </div>
        )}
      </motion.div>

      {/* 总结 */}
      <div className="mt-6 bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg border-2 border-green-300">
        <h4 className="font-bold text-green-800 mb-3">关键发现</h4>
        <ul className="text-sm text-slate-700 space-y-2">
          <li>✓ 速度提升：<strong>2.3-2.8倍</strong>（接近理论上限）</li>
          <li>✓ 显存节省：<strong>~50%</strong>（权重+激活值都减半）</li>
          <li>✓ 准确率影响：<strong>&lt;0.1%</strong>（BF16 几乎无损）</li>
          <li>✓ 超大模型（&gt;7B）：FP32 单卡 OOM，<strong>BF16 必需</strong></li>
        </ul>
      </div>
    </div>
  )
}

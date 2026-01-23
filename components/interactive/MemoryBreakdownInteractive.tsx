'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Database, Cpu, Zap, TrendingUp } from 'lucide-react'

export default function MemoryBreakdownInteractive() {
  const [modelSize, setModelSize] = useState<'7B' | '13B' | '70B'>('7B')
  const [batchSize, setBatchSize] = useState(1)
  const [seqLength, setSeqLength] = useState(512)

  // 显存计算（简化公式）
  const calculateMemory = () => {
    const paramCount = {
      '7B': 7,
      '13B': 13,
      '70B': 70,
    }[modelSize]

    const weights = paramCount * 2 // FP16: 2 bytes/param
    const optimizer = weights * 2 // AdamW: momentum + variance
    const gradients = weights // Same size as weights
    const activations = (paramCount / 7) * batchSize * (seqLength / 512) * 2.5 // 估算

    return {
      weights,
      optimizer,
      gradients,
      activations,
      total: weights + optimizer + gradients + activations,
    }
  }

  const memory = calculateMemory()
  const components = [
    {
      name: '模型权重',
      value: memory.weights,
      color: 'from-blue-500 to-blue-600',
      icon: Database,
      description: `${modelSize} 参数 × 2 bytes (FP16)`,
    },
    {
      name: '优化器状态',
      value: memory.optimizer,
      color: 'from-purple-500 to-purple-600',
      icon: Cpu,
      description: 'Momentum + Variance (AdamW)',
    },
    {
      name: '梯度',
      value: memory.gradients,
      color: 'from-green-500 to-green-600',
      icon: TrendingUp,
      description: '反向传播时的梯度',
    },
    {
      name: '激活值',
      value: memory.activations,
      color: 'from-amber-500 to-amber-600',
      icon: Zap,
      description: '前向传播的中间结果',
    },
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        训练显存占用分析
      </h3>

      {/* 参数控制 */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg border border-slate-200">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            模型大小
          </label>
          <div className="flex gap-2">
            {(['7B', '13B', '70B'] as const).map((size) => (
              <button
                key={size}
                onClick={() => setModelSize(size)}
                className={`flex-1 py-2 px-3 rounded-lg border-2 transition-all ${
                  modelSize === size
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-bold'
                    : 'border-slate-300 hover:border-blue-300'
                }`}
              >
                {size}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-slate-200">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Batch Size: {batchSize}
          </label>
          <input
            type="range"
            min="1"
            max="32"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="bg-white p-4 rounded-lg border border-slate-200">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Seq Length: {seqLength}
          </label>
          <input
            type="range"
            min="128"
            max="4096"
            step="128"
            value={seqLength}
            onChange={(e) => setSeqLength(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* 显存占用饼图 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-slate-800">显存组成</h4>
          <div className="text-2xl font-bold text-blue-600">
            {memory.total.toFixed(1)} GB
          </div>
        </div>

        {/* 堆叠条形图 */}
        <div className="h-16 flex rounded-lg overflow-hidden mb-4">
          {components.map((comp, idx) => (
            <motion.div
              key={idx}
              className={`bg-gradient-to-r ${comp.color} flex items-center justify-center text-white text-sm font-bold relative group`}
              initial={{ width: 0 }}
              animate={{ width: `${(comp.value / memory.total) * 100}%` }}
              transition={{ duration: 0.8, delay: idx * 0.1 }}
            >
              <span className="opacity-0 group-hover:opacity-100 transition-opacity">
                {comp.value.toFixed(1)} GB
              </span>
              <div className="absolute bottom-full mb-2 hidden group-hover:block bg-slate-800 text-white px-3 py-2 rounded text-xs whitespace-nowrap z-10">
                <div className="font-bold">{comp.name}</div>
                <div className="opacity-75">{comp.description}</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* 图例 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {components.map((comp, idx) => {
            const Icon = comp.icon
            const percentage = ((comp.value / memory.total) * 100).toFixed(1)
            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 + 0.5 }}
                className="flex items-center gap-2"
              >
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${comp.color} flex items-center justify-center`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-slate-600 truncate">
                    {comp.name}
                  </div>
                  <div className="text-sm font-bold text-slate-800">
                    {comp.value.toFixed(1)} GB ({percentage}%)
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* 优化建议 */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg border-2 ${
          memory.total > 80 ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'
        }`}>
          <h4 className={`font-bold mb-2 ${
            memory.total > 80 ? 'text-red-800' : 'text-green-800'
          }`}>
            {memory.total > 80 ? '⚠️ 显存不足' : '✅ 显存充足'}
          </h4>
          <div className={`text-sm ${
            memory.total > 80 ? 'text-red-700' : 'text-green-700'
          }`}>
            {memory.total > 80 ? (
              <ul className="space-y-1">
                <li>• 启用 Gradient Checkpointing (-30%)</li>
                <li>• 使用 QLoRA 4-bit (-75%)</li>
                <li>• 减小 Batch Size 或 Seq Length</li>
                <li>• 使用 DeepSpeed ZeRO-3</li>
              </ul>
            ) : (
              <ul className="space-y-1">
                <li>• 当前配置可以正常训练</li>
                <li>• 可尝试增大 Batch Size 提升效率</li>
                <li>• 或增加序列长度训练长文本</li>
              </ul>
            )}
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
          <h4 className="font-bold text-blue-800 mb-2">💡 激活值优化</h4>
          <div className="text-sm text-blue-700 space-y-1">
            <div>当前: {memory.activations.toFixed(1)} GB</div>
            <div className="border-t border-blue-200 my-2"></div>
            <div>✓ Gradient Checkpointing: {(memory.activations * 0.35).toFixed(1)} GB (-65%)</div>
            <div>✓ + Flash Attention: {(memory.activations * 0.25).toFixed(1)} GB (-75%)</div>
          </div>
        </div>
      </div>

      {/* 公式 */}
      <div className="mt-4 bg-white p-4 rounded-lg border border-slate-200">
        <h4 className="font-bold text-slate-800 mb-2">显存计算公式</h4>
        <div className="text-sm text-slate-600 space-y-1 font-mono">
          <div>权重 = N × 2 bytes (FP16)</div>
          <div>优化器 = N × 4 bytes (AdamW: momentum + variance)</div>
          <div>梯度 = N × 2 bytes</div>
          <div>激活值 = O(L × B × S × d) (L=层数, B=batch, S=序列长度, d=维度)</div>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 调整参数查看显存变化 | 悬停条形图查看详情
      </div>
    </div>
  )
}

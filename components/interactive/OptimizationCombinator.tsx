'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, CheckCircle2, TrendingDown } from 'lucide-react'

type Optimization = 'checkpoint' | 'flash' | 'accumulation' | 'qlora' | 'zero3'

export default function OptimizationCombinator() {
  const [selectedOptimizations, setSelectedOptimizations] = useState<Set<Optimization>>(
    new Set(['checkpoint'])
  )

  const baseMemory = {
    weights: 14, // GB (LLaMA-7B FP16)
    optimizer: 28,
    gradients: 14,
    activations: 20,
  }

  const optimizations = {
    checkpoint: {
      name: 'Gradient Checkpointing',
      icon: '🔄',
      color: 'blue',
      effect: { activations: 0.35 }, // 减少 65% 激活值显存
      speedImpact: -25, // 速度降低 25%
      description: '通过重计算减少激活值显存',
    },
    flash: {
      name: 'Flash Attention',
      icon: '⚡',
      color: 'amber',
      effect: { activations: 0.7 }, // 在 checkpoint 基础上再减少 30%
      speedImpact: +10, // 速度提升 10%
      description: 'IO-aware 算法，减少内存访问',
      requires: 'checkpoint',
    },
    accumulation: {
      name: 'Gradient Accumulation',
      icon: '📊',
      color: 'green',
      effect: { activations: 0.5 }, // 可以减半激活值（通过减小 micro-batch）
      speedImpact: -5,
      description: '用时间换空间，累积梯度',
    },
    qlora: {
      name: 'QLoRA (4-bit)',
      icon: '🎯',
      color: 'purple',
      effect: { weights: 0.25, optimizer: 0.1, gradients: 0.25 }, // 4-bit 量化
      speedImpact: -10,
      description: '4-bit 量化 + LoRA 微调',
    },
    zero3: {
      name: 'DeepSpeed ZeRO-3 Offload',
      icon: '🚀',
      color: 'red',
      effect: { weights: 0.1, optimizer: 0, gradients: 0.1 }, // Offload 到 CPU
      speedImpact: -60,
      description: 'CPU/NVMe offload',
    },
  }

  const toggleOptimization = (opt: Optimization) => {
    const newSet = new Set(selectedOptimizations)
    if (newSet.has(opt)) {
      newSet.delete(opt)
      // 删除依赖项
      if (opt === 'checkpoint' && newSet.has('flash')) {
        newSet.delete('flash')
      }
    } else {
      newSet.add(opt)
      // 自动添加依赖
      if (opt === 'flash' && !newSet.has('checkpoint')) {
        newSet.add('checkpoint')
      }
    }
    setSelectedOptimizations(newSet)
  }

  const calculateMemory = () => {
    let memory = { ...baseMemory }
    let speedMultiplier = 100

    // 应用优化
    selectedOptimizations.forEach((opt) => {
      const optConfig = optimizations[opt]
      Object.entries(optConfig.effect).forEach(([key, multiplier]) => {
        memory[key as keyof typeof memory] *= multiplier
      })
      speedMultiplier += optConfig.speedImpact
    })

    const total = Object.values(memory).reduce((sum, val) => sum + val, 0)
    const reduction = ((baseMemory.weights + baseMemory.optimizer + baseMemory.gradients + baseMemory.activations - total) /
      (baseMemory.weights + baseMemory.optimizer + baseMemory.gradients + baseMemory.activations)) * 100

    return { ...memory, total, reduction, speedMultiplier }
  }

  const memory = calculateMemory()
  const baseTotal = baseMemory.weights + baseMemory.optimizer + baseMemory.gradients + baseMemory.activations

  const memoryComponents = [
    { name: '权重', value: memory.weights, color: 'from-blue-400 to-blue-600' },
    { name: '优化器', value: memory.optimizer, color: 'from-purple-400 to-purple-600' },
    { name: '梯度', value: memory.gradients, color: 'from-green-400 to-green-600' },
    { name: '激活值', value: memory.activations, color: 'from-amber-400 to-amber-600' },
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        内存优化组合器
      </h3>

      {/* 优化选项 */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
        {(Object.entries(optimizations) as [Optimization, typeof optimizations[Optimization]][]).map(([key, opt]) => {
          const isSelected = selectedOptimizations.has(key)
          const isDisabled = (opt as any).requires && !selectedOptimizations.has((opt as any).requires as Optimization)
          
          return (
            <motion.button
              key={key}
              onClick={() => !isDisabled && toggleOptimization(key)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                isDisabled
                  ? 'border-slate-200 bg-slate-100 opacity-50 cursor-not-allowed'
                  : isSelected
                  ? `border-${opt.color}-500 bg-${opt.color}-50`
                  : 'border-slate-300 bg-white hover:border-' + opt.color + '-300'
              }`}
              whileHover={!isDisabled ? { scale: 1.02 } : {}}
              whileTap={!isDisabled ? { scale: 0.98 } : {}}
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{opt.icon}</span>
                <div className="flex-1">
                  <div className="font-bold text-slate-800">{opt.name}</div>
                  {isSelected && (
                    <CheckCircle2 className={`w-4 h-4 text-${opt.color}-600 mt-1`} />
                  )}
                </div>
              </div>
              <div className="text-xs text-slate-600">{opt.description}</div>
              {opt.speedImpact !== 0 && (
                <div className={`text-xs mt-2 font-medium ${
                  opt.speedImpact > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  速度: {opt.speedImpact > 0 ? '+' : ''}{opt.speedImpact}%
                </div>
              )}
              {(opt as any).requires && (
                <div className="text-xs text-amber-600 mt-1">
                  需要: {optimizations[(opt as any).requires as Optimization].name}
                </div>
              )}
            </motion.button>
          )
        })}
      </div>

      {/* 显存占用对比 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-slate-800">显存占用分析</h4>
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600">
              {memory.total.toFixed(1)} GB
            </div>
            <div className="text-sm text-green-600">
              节省 {memory.reduction.toFixed(1)}%
            </div>
          </div>
        </div>

        {/* 对比条形图 */}
        <div className="space-y-3 mb-4">
          <div>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-slate-600">基准配置</span>
              <span className="text-slate-700 font-bold">{baseTotal.toFixed(1)} GB</span>
            </div>
            <div className="h-8 bg-slate-100 rounded-lg overflow-hidden">
              <div className="h-full bg-gradient-to-r from-slate-400 to-slate-600 flex items-center justify-center text-white text-sm font-bold">
                100%
              </div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-slate-600">当前配置</span>
              <span className="text-blue-700 font-bold">{memory.total.toFixed(1)} GB</span>
            </div>
            <div className="h-8 bg-slate-100 rounded-lg overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center text-white text-sm font-bold"
                initial={{ width: 0 }}
                animate={{ width: `${(memory.total / baseTotal) * 100}%` }}
                transition={{ duration: 0.8 }}
              >
                {((memory.total / baseTotal) * 100).toFixed(0)}%
              </motion.div>
            </div>
          </div>
        </div>

        {/* 显存组成 */}
        <div className="space-y-2">
          {memoryComponents.map((comp, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="flex items-center gap-3"
            >
              <div className="w-20 text-sm text-slate-600">{comp.name}</div>
              <div className="flex-1 h-6 bg-slate-100 rounded overflow-hidden">
                <motion.div
                  className={`h-full bg-gradient-to-r ${comp.color} flex items-center justify-center text-white text-xs font-bold`}
                  initial={{ width: 0 }}
                  animate={{ width: `${(comp.value / baseTotal) * 100}%` }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                >
                  {comp.value.toFixed(1)} GB
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 性能影响 */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
          <h4 className="font-bold text-green-800 mb-3 flex items-center gap-2">
            <TrendingDown className="w-5 h-5" />
            显存节省
          </h4>
          <div className="text-3xl font-bold text-green-700 mb-2">
            {memory.reduction.toFixed(1)}%
          </div>
          <div className="text-sm text-green-600">
            从 {baseTotal.toFixed(1)} GB → {memory.total.toFixed(1)} GB
          </div>
        </div>

        <div className={`bg-gradient-to-br p-4 rounded-lg border-2 ${
          memory.speedMultiplier >= 80
            ? 'from-green-50 to-emerald-50 border-green-200'
            : memory.speedMultiplier >= 50
            ? 'from-amber-50 to-orange-50 border-amber-200'
            : 'from-red-50 to-rose-50 border-red-200'
        }`}>
          <h4 className={`font-bold mb-3 flex items-center gap-2 ${
            memory.speedMultiplier >= 80 ? 'text-green-800' :
            memory.speedMultiplier >= 50 ? 'text-amber-800' : 'text-red-800'
          }`}>
            <Zap className="w-5 h-5" />
            训练速度
          </h4>
          <div className={`text-3xl font-bold mb-2 ${
            memory.speedMultiplier >= 80 ? 'text-green-700' :
            memory.speedMultiplier >= 50 ? 'text-amber-700' : 'text-red-700'
          }`}>
            {memory.speedMultiplier}%
          </div>
          <div className={`text-sm ${
            memory.speedMultiplier >= 80 ? 'text-green-600' :
            memory.speedMultiplier >= 50 ? 'text-amber-600' : 'text-red-600'
          }`}>
            {memory.speedMultiplier > 100 ? '+' : ''}{memory.speedMultiplier - 100}% 相对基准
          </div>
        </div>
      </div>

      {/* 建议 */}
      <div className="mt-4 bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
        <h4 className="font-bold text-blue-800 mb-2">💡 推荐组合</h4>
        <div className="text-sm text-blue-700 space-y-1">
          {memory.total > 80 && (
            <div>• 显存仍然过高，建议添加 QLoRA 或 ZeRO-3</div>
          )}
          {memory.total <= 24 && memory.total > 16 && (
            <div>• ✅ 适合 RTX 3090/4090 (24GB)</div>
          )}
          {memory.total <= 16 && (
            <div>• ✅ 适合 RTX 3060 (12GB) 及以上</div>
          )}
          {memory.speedMultiplier < 50 && (
            <div>• ⚠️ 训练速度较慢，考虑移除 ZeRO-3 或使用更少优化</div>
          )}
          {!selectedOptimizations.has('checkpoint') && memory.total > 40 && (
            <div>• 建议启用 Gradient Checkpointing</div>
          )}
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        🎯 点击卡片组合优化策略 | 基于 LLaMA-7B 测算
      </div>
    </div>
  )
}

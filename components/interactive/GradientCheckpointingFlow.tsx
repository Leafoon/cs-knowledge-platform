'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, ChevronRight } from 'lucide-react'

export default function GradientCheckpointingFlow() {
  const [mode, setMode] = useState<'normal' | 'checkpoint'>('normal')
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const layers = Array.from({ length: 8 }, (_, i) => ({
    id: i,
    name: `Layer ${i}`,
  }))

  const steps = mode === 'normal' ? [
    { phase: 'forward', desc: '前向传播：计算并保存所有激活值', activeLayer: 0, savedActivations: [0] },
    { phase: 'forward', desc: '计算 Layer 1', activeLayer: 1, savedActivations: [0, 1] },
    { phase: 'forward', desc: '计算 Layer 2-7', activeLayer: 7, savedActivations: [0, 1, 2, 3, 4, 5, 6, 7] },
    { phase: 'backward', desc: '反向传播：使用保存的激活值计算梯度', activeLayer: 7, savedActivations: [0, 1, 2, 3, 4, 5, 6, 7] },
    { phase: 'backward', desc: '反向到 Layer 0', activeLayer: 0, savedActivations: [0, 1, 2, 3, 4, 5, 6, 7] },
  ] : [
    { phase: 'forward', desc: '前向传播：只保存检查点（每2层）', activeLayer: 0, savedActivations: [0], checkpoints: [0] },
    { phase: 'forward', desc: '计算 Layer 1（不保存）', activeLayer: 1, savedActivations: [0], checkpoints: [0] },
    { phase: 'forward', desc: '保存 Layer 2 检查点', activeLayer: 2, savedActivations: [0, 2], checkpoints: [0, 2] },
    { phase: 'forward', desc: '计算完成，仅保存检查点', activeLayer: 7, savedActivations: [0, 2, 4, 6], checkpoints: [0, 2, 4, 6] },
    { phase: 'backward', desc: '反向传播：从 Layer 6 重新计算 Layer 7', activeLayer: 7, savedActivations: [0, 2, 4, 6, 7], checkpoints: [0, 2, 4, 6], recomputing: [7] },
    { phase: 'backward', desc: '从 Layer 4 重新计算 Layer 5-6', activeLayer: 5, savedActivations: [0, 2, 4, 5, 6], checkpoints: [0, 2, 4, 6], recomputing: [5, 6] },
    { phase: 'backward', desc: '完成反向传播', activeLayer: 0, savedActivations: [0, 2, 4, 6], checkpoints: [0, 2, 4, 6] },
  ]

  React.useEffect(() => {
    if (!isPlaying) return
    const timer = setInterval(() => {
      setStep((s) => {
        if (s >= steps.length - 1) {
          setIsPlaying(false)
          return s
        }
        return s + 1
      })
    }, 1500)
    return () => clearInterval(timer)
  }, [isPlaying, steps.length])

  const currentStep = steps[step]
  const memoryUsage = mode === 'normal'
    ? layers.length
    : (currentStep.savedActivations?.length || 0)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        Gradient Checkpointing 流程对比
      </h3>

      {/* 模式选择 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={() => { setMode('normal'); setStep(0); setIsPlaying(false) }}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'normal'
              ? 'border-blue-500 bg-blue-50'
              : 'border-slate-300 bg-white hover:border-blue-300'
          }`}
        >
          <div className="font-bold text-lg mb-1">标准反向传播</div>
          <div className="text-sm text-slate-600">保存所有激活值</div>
          <div className="mt-2 text-xs text-red-600 font-bold">显存: 100%</div>
        </button>

        <button
          onClick={() => { setMode('checkpoint'); setStep(0); setIsPlaying(false) }}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            mode === 'checkpoint'
              ? 'border-green-500 bg-green-50'
              : 'border-slate-300 bg-white hover:border-green-300'
          }`}
        >
          <div className="font-bold text-lg mb-1">Gradient Checkpointing</div>
          <div className="text-sm text-slate-600">只保存检查点</div>
          <div className="mt-2 text-xs text-green-600 font-bold">显存: ~35%</div>
        </button>
      </div>

      {/* 显存占用实时显示 */}
      <div className="bg-white p-4 rounded-lg border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-600">显存占用（激活值）</span>
          <span className="text-sm font-bold text-blue-600">
            {memoryUsage} / {layers.length} 层 ({((memoryUsage / layers.length) * 100).toFixed(0)}%)
          </span>
        </div>
        <div className="h-6 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            className={`h-full ${
              mode === 'normal' ? 'bg-gradient-to-r from-red-500 to-red-600' : 'bg-gradient-to-r from-green-500 to-green-600'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${(memoryUsage / layers.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* 层可视化 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="mb-4">
          <h4 className="font-bold text-slate-800 mb-2">{currentStep.desc}</h4>
          <div className={`text-sm font-medium ${
            currentStep.phase === 'forward' ? 'text-blue-600' : 'text-purple-600'
          }`}>
            {currentStep.phase === 'forward' ? '→ 前向传播' : '← 反向传播'}
          </div>
        </div>

        <div className="flex gap-2 mb-4">
          {layers.map((layer) => {
            const isSaved = currentStep.savedActivations?.includes(layer.id)
            const isCheckpoint = (currentStep as any).checkpoints?.includes(layer.id)
            const isRecomputing = (currentStep as any).recomputing?.includes(layer.id)
            const isActive = currentStep.activeLayer === layer.id

            return (
              <motion.div
                key={layer.id}
                className={`flex-1 h-24 rounded-lg border-2 flex flex-col items-center justify-center relative ${
                  isActive
                    ? 'border-yellow-500 bg-yellow-100'
                    : isSaved
                    ? isCheckpoint
                      ? 'border-green-500 bg-green-100'
                      : 'border-blue-500 bg-blue-100'
                    : 'border-slate-300 bg-slate-50'
                }`}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: layer.id * 0.05 }}
              >
                <div className="text-xs font-bold text-slate-700">{layer.name}</div>
                {isSaved && (
                  <div className={`text-xs mt-1 font-medium ${
                    isCheckpoint ? 'text-green-700' : 'text-blue-700'
                  }`}>
                    {isCheckpoint ? '🔖 检查点' : '💾 已保存'}
                  </div>
                )}
                {isRecomputing && (
                  <motion.div
                    className="absolute inset-0 bg-yellow-200/50 rounded-lg flex items-center justify-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ repeat: Infinity, duration: 1, repeatType: 'reverse' }}
                  >
                    <span className="text-xs font-bold text-yellow-800">🔄 重计算</span>
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </div>

        {/* 图例 */}
        <div className="flex gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded border-2 border-yellow-500 bg-yellow-100"></div>
            <span className="text-slate-600">当前计算</span>
          </div>
          {mode === 'checkpoint' && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded border-2 border-green-500 bg-green-100"></div>
                <span className="text-slate-600">检查点</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-yellow-200/50"></div>
                <span className="text-slate-600">重计算中</span>
              </div>
            </>
          )}
          {mode === 'normal' && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded border-2 border-blue-500 bg-blue-100"></div>
              <span className="text-slate-600">已保存</span>
            </div>
          )}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex items-center justify-center gap-4 mb-6">
        <button
          onClick={() => { setStep(0); setIsPlaying(false) }}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 rounded-lg flex items-center gap-2 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-6 py-2 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          {isPlaying ? (
            <>
              <Pause className="w-4 h-4" />
              暂停
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              播放
            </>
          )}
        </button>

        <button
          onClick={() => setStep((s) => Math.min(s + 1, steps.length - 1))}
          disabled={step >= steps.length - 1}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* 对比总结 */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-red-50 to-orange-50 p-4 rounded-lg border border-red-200">
          <h4 className="font-bold text-red-800 mb-2">标准反向传播</h4>
          <ul className="text-sm text-red-700 space-y-1">
            <li>✓ 速度快（无重计算）</li>
            <li>✓ 实现简单</li>
            <li>✗ 显存占用高（100%）</li>
            <li>✗ 限制 batch size</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
          <h4 className="font-bold text-green-800 mb-2">Gradient Checkpointing</h4>
          <ul className="text-sm text-green-700 space-y-1">
            <li>✓ 显存节省 65%</li>
            <li>✓ 支持更大 batch</li>
            <li>✗ 速度慢 25-30%</li>
            <li>✗ 需要重计算激活值</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        🎯 步骤 {step + 1} / {steps.length} | 观察显存占用变化
      </div>
    </div>
  )
}

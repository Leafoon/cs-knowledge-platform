'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, ChevronRight } from 'lucide-react'

export default function GPTQAlgorithmFlow() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const steps = [
    {
      title: '初始状态',
      description: '加载预训练权重矩阵 W 和校准数据 X',
      formula: '\\mathbf{W} \\in \\mathbb{R}^{d_{out} \\times d_{in}}',
      visual: {
        type: 'matrix',
        data: Array.from({ length: 4 }, () =>
          Array.from({ length: 8 }, () => Math.random() * 2 - 1)
        ),
        quantized: [],
      },
    },
    {
      title: '计算 Hessian 矩阵',
      description: '基于校准数据计算二阶统计信息',
      formula: '\\mathbf{H} = 2\\mathbf{X}\\mathbf{X}^T',
      visual: {
        type: 'hessian',
        highlight: 'diagonal',
      },
    },
    {
      title: 'Cholesky 分解',
      description: '将 Hessian 矩阵分解为下三角矩阵',
      formula: '\\mathbf{H} = \\mathbf{L}\\mathbf{L}^T',
      visual: {
        type: 'decomposition',
      },
    },
    {
      title: '逐列量化 (第1列)',
      description: '量化第一列权重，误差传播到未量化列',
      formula: 'w_q[i] = \\text{quant}(w[i]), \\quad w[i+1:] -= \\text{error} \\cdot (\\mathbf{H}[i, i+1:] / \\mathbf{H}[i, i])',
      visual: {
        type: 'matrix',
        data: Array.from({ length: 4 }, () =>
          Array.from({ length: 8 }, () => Math.random() * 2 - 1)
        ),
        quantized: [0],
      },
    },
    {
      title: '逐列量化 (第2-4列)',
      description: '继续量化，误差持续补偿',
      formula: '\\text{for } i = 1, 2, \\ldots, d_{in}',
      visual: {
        type: 'matrix',
        data: Array.from({ length: 4 }, () =>
          Array.from({ length: 8 }, () => Math.random() * 2 - 1)
        ),
        quantized: [0, 1, 2, 3],
      },
    },
    {
      title: '分组量化 (group_size=4)',
      description: '每4列共享一个 scale，减少量化参数',
      formula: 's_g = \\frac{\\max(\\mathbf{W}_{[:, g \\cdot k:(g+1) \\cdot k]}) - \\min(\\mathbf{W}_{[:, g \\cdot k:(g+1) \\cdot k]})}{15}',
      visual: {
        type: 'grouped',
        groupSize: 4,
      },
    },
    {
      title: '完成',
      description: '所有权重量化完成，保存量化模型',
      formula: '\\mathbf{W}_q \\approx \\mathbf{W}, \\quad \\text{size} = \\frac{1}{8} \\times \\text{original}',
      visual: {
        type: 'matrix',
        data: Array.from({ length: 4 }, () =>
          Array.from({ length: 8 }, () => Math.random() * 2 - 1)
        ),
        quantized: [0, 1, 2, 3, 4, 5, 6, 7],
      },
    },
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
    }, 2000)
    return () => clearInterval(timer)
  }, [isPlaying, steps.length])

  const currentStep = steps[step]

  const renderVisual = () => {
    if (currentStep.visual.type === 'matrix') {
      const { data, quantized } = currentStep.visual
      return (
        <div className="grid gap-1">
          {data?.map((row, rowIdx) => (
            <div key={rowIdx} className="flex gap-1">
              {row.map((val, colIdx) => {
                const isQuantized = quantized?.includes(colIdx)
                return (
                  <motion.div
                    key={colIdx}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: colIdx * 0.05 }}
                    className={`flex-1 h-10 rounded flex items-center justify-center text-xs font-mono ${
                      isQuantized
                        ? 'bg-gradient-to-br from-green-400 to-green-600 text-white'
                        : 'bg-gradient-to-br from-blue-400 to-blue-600 text-white'
                    }`}
                  >
                    {isQuantized ? Math.round(val * 7) : val.toFixed(2)}
                  </motion.div>
                )
              })}
            </div>
          ))}
        </div>
      )
    }

    if (currentStep.visual.type === 'hessian') {
      return (
        <div className="grid grid-cols-8 gap-1">
          {Array.from({ length: 64 }, (_, i) => {
            const row = Math.floor(i / 8)
            const col = i % 8
            const isDiagonal = row === col
            const value = isDiagonal ? 1 : Math.random() * 0.3
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.01 }}
                className={`h-10 rounded flex items-center justify-center text-xs font-mono ${
                  isDiagonal
                    ? 'bg-gradient-to-br from-red-400 to-red-600 text-white font-bold'
                    : 'bg-slate-200 text-slate-600'
                }`}
              >
                {value.toFixed(2)}
              </motion.div>
            )
          })}
        </div>
      )
    }

    if (currentStep.visual.type === 'decomposition') {
      return (
        <div className="flex items-center gap-4">
          <div className="flex-1 p-4 bg-blue-100 rounded-lg">
            <div className="font-bold text-blue-800 mb-2">H (Hessian)</div>
            <div className="grid grid-cols-4 gap-1">
              {Array.from({ length: 16 }, (_, i) => (
                <div key={i} className="h-8 bg-blue-300 rounded" />
              ))}
            </div>
          </div>
          <ChevronRight className="w-6 h-6 text-slate-400" />
          <div className="flex-1 p-4 bg-green-100 rounded-lg">
            <div className="font-bold text-green-800 mb-2">L (下三角)</div>
            <div className="grid grid-cols-4 gap-1">
              {Array.from({ length: 16 }, (_, i) => {
                const row = Math.floor(i / 4)
                const col = i % 4
                const isLower = col <= row
                return (
                  <div
                    key={i}
                    className={`h-8 rounded ${
                      isLower ? 'bg-green-400' : 'bg-slate-200'
                    }`}
                  />
                )
              })}
            </div>
          </div>
        </div>
      )
    }

    if (currentStep.visual.type === 'grouped') {
      const { groupSize } = currentStep.visual
      const colors = ['from-red-400 to-red-600', 'from-blue-400 to-blue-600']
      return (
        <div className="grid gap-1">
          {Array.from({ length: 4 }, (_, rowIdx) => (
            <div key={rowIdx} className="flex gap-1">
              {Array.from({ length: 8 }, (_, colIdx) => {
                const groupId = Math.floor(colIdx / (groupSize || 1))
                return (
                  <div
                    key={colIdx}
                    className={`flex-1 h-10 rounded bg-gradient-to-br ${colors[groupId]} flex items-center justify-center text-white text-xs font-bold`}
                  >
                    G{groupId}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      )
    }

    return null
  }

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        GPTQ 算法流程
      </h3>

      {/* 进度条 */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-600">
            步骤 {step + 1} / {steps.length}
          </span>
          <span className="text-xs text-slate-500">
            {Math.round(((step + 1) / steps.length) * 100)}%
          </span>
        </div>
        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-green-500 to-blue-500"
            initial={{ width: 0 }}
            animate={{ width: `${((step + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* 当前步骤 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="bg-white p-6 rounded-xl border border-slate-200 mb-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-blue-500 flex items-center justify-center text-white font-bold">
              {step + 1}
            </div>
            <div>
              <h4 className="font-bold text-lg text-slate-800">{currentStep.title}</h4>
              <p className="text-sm text-slate-600">{currentStep.description}</p>
            </div>
          </div>

          {/* 数学公式 */}
          <div className="bg-blue-50 p-4 rounded-lg mb-4 overflow-x-auto">
            <div className="font-mono text-sm text-center">
              ${currentStep.formula}$
            </div>
          </div>

          {/* 可视化 */}
          <div className="bg-slate-50 p-4 rounded-lg">
            {renderVisual()}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 控制按钮 */}
      <div className="flex items-center justify-center gap-4">
        <motion.button
          onClick={() => {
            setStep(0)
            setIsPlaying(false)
          }}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 rounded-lg flex items-center gap-2 transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </motion.button>

        <motion.button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-6 py-2 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white rounded-lg flex items-center gap-2 transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
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
        </motion.button>

        <motion.button
          onClick={() => setStep((s) => Math.min(s + 1, steps.length - 1))}
          disabled={step >= steps.length - 1}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          whileHover={{ scale: step < steps.length - 1 ? 1.05 : 1 }}
          whileTap={{ scale: step < steps.length - 1 ? 0.95 : 1 }}
        >
          下一步
          <ChevronRight className="w-4 h-4" />
        </motion.button>
      </div>

      <div className="mt-6 bg-gradient-to-r from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-200">
        <div className="font-bold text-amber-800 mb-2">🔑 关键洞察</div>
        <ul className="text-sm text-amber-700 space-y-1">
          <li>• Hessian 对角线大 → 该权重更重要 → 量化更谨慎</li>
          <li>• 误差传播通过 H⁻¹ 补偿到未量化权重</li>
          <li>• 分组量化 (group_size=128) 平衡精度与内存</li>
        </ul>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        🎯 绿色方块 = 已量化 | 蓝色方块 = 未量化
      </div>
    </div>
  )
}

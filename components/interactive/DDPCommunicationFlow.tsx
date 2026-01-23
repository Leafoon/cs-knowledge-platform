'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Cpu } from 'lucide-react'

interface GPUNode {
  id: number
  gradient: number[]
  color: string
}

export default function DDPCommunicationFlow() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [gpus, setGpus] = useState<GPUNode[]>([
    { id: 0, gradient: [1, 2, 3, 4], color: 'from-blue-400 to-blue-600' },
    { id: 1, gradient: [5, 6, 7, 8], color: 'from-purple-400 to-purple-600' },
    { id: 2, gradient: [9, 10, 11, 12], color: 'from-green-400 to-green-600' },
    { id: 3, gradient: [13, 14, 15, 16], color: 'from-orange-400 to-orange-600' },
  ])

  const maxSteps = 7 // Reduce-Scatter (3) + AllGather (3) + Final (1)

  useEffect(() => {
    if (isPlaying && step < maxSteps) {
      const timer = setTimeout(() => {
        setStep(step + 1)
      }, 1500)
      return () => clearTimeout(timer)
    } else if (step >= maxSteps) {
      setIsPlaying(false)
    }
  }, [isPlaying, step])

  const reset = () => {
    setStep(0)
    setIsPlaying(false)
    setGpus([
      { id: 0, gradient: [1, 2, 3, 4], color: 'from-blue-400 to-blue-600' },
      { id: 1, gradient: [5, 6, 7, 8], color: 'from-purple-400 to-purple-600' },
      { id: 2, gradient: [9, 10, 11, 12], color: 'from-green-400 to-green-600' },
      { id: 3, gradient: [13, 14, 15, 16], color: 'from-orange-400 to-orange-600' },
    ])
  }

  // Ring-AllReduce步骤说明
  const getStepDescription = () => {
    if (step === 0) return '初始状态：每个GPU持有本地梯度'
    if (step <= 3) return `Reduce-Scatter 阶段 - 步骤 ${step}/3：环形发送和累加梯度块`
    if (step <= 6) return `AllGather 阶段 - 步骤 ${step - 3}/3：收集完整的平均梯度`
    return '完成：所有GPU持有相同的平均梯度'
  }

  // 模拟Reduce-Scatter和AllGather
  const getGradientAtStep = (gpuId: number, chunkIdx: number) => {
    const originalGrads = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ]

    // Reduce-Scatter阶段
    if (step >= 1 && step <= 3) {
      // 在对应chunk累加
      const targetChunk = (gpuId + 4 - step) % 4
      if (chunkIdx === targetChunk) {
        // 累加了step个GPU的值
        let sum = 0
        for (let i = 0; i <= step; i++) {
          const sourceGpu = (gpuId + 4 - i) % 4
          sum += originalGrads[sourceGpu][chunkIdx]
        }
        return sum
      }
      return originalGrads[gpuId][chunkIdx]
    }

    // AllGather阶段 & 完成
    if (step >= 4) {
      // 所有chunk都是平均值
      let sum = 0
      for (let i = 0; i < 4; i++) {
        sum += originalGrads[i][chunkIdx]
      }
      return sum / 4
    }

    return originalGrads[gpuId][chunkIdx]
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Cpu className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-slate-800">DDP Ring-AllReduce 通信流程</h3>
        </div>

        {/* 控制按钮 */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? '暂停' : '播放'}
          </button>
          <button
            onClick={reset}
            className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </button>
        </div>
      </div>

      {/* 步骤说明 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-600">当前步骤</div>
            <div className="text-lg font-bold text-indigo-600">{getStepDescription()}</div>
          </div>
          <div className="text-3xl font-bold text-slate-800">
            {step}/{maxSteps}
          </div>
        </div>
      </div>

      {/* GPU可视化 */}
      <div className="grid grid-cols-4 gap-6 mb-6">
        {gpus.map((gpu) => (
          <motion.div
            key={gpu.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: gpu.id * 0.1 }}
            className="bg-white p-4 rounded-lg shadow-lg"
          >
            <div className={`w-full h-24 bg-gradient-to-br ${gpu.color} rounded-lg shadow-md flex items-center justify-center text-white font-bold text-xl mb-3`}>
              GPU {gpu.id}
            </div>

            {/* 梯度块 */}
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-600 mb-2">梯度块</div>
              {[0, 1, 2, 3].map((chunkIdx) => {
                const value = getGradientAtStep(gpu.id, chunkIdx)
                const isAvgValue = value === Math.floor(value + 0.5) && value > 16
                
                return (
                  <motion.div
                    key={chunkIdx}
                    layout
                    className={`p-2 rounded text-center font-mono text-sm ${
                      isAvgValue
                        ? 'bg-green-100 border-2 border-green-400'
                        : 'bg-slate-100 border border-slate-300'
                    }`}
                  >
                    <AnimatePresence mode="wait">
                      <motion.span
                        key={`${gpu.id}-${chunkIdx}-${step}`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{ duration: 0.3 }}
                        className={isAvgValue ? 'text-green-700 font-bold' : 'text-slate-700'}
                      >
                        {value.toFixed(1)}
                      </motion.span>
                    </AnimatePresence>
                  </motion.div>
                )
              })}
            </div>
          </motion.div>
        ))}
      </div>

      {/* 通信箭头提示 */}
      {step > 0 && step < maxSteps && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg"
        >
          <div className="flex items-center gap-2 text-blue-800">
            <div className="text-2xl">🔄</div>
            <div>
              <div className="font-bold">
                {step <= 3 ? 'Reduce-Scatter' : 'AllGather'}
              </div>
              <div className="text-sm">
                {step <= 3
                  ? '每个GPU向右侧邻居发送梯度块，并累加接收到的块'
                  : '每个GPU向右侧邻居发送已归约的块，收集完整梯度'}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 算法说明 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-3">Ring-AllReduce 优势</h4>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="p-3 bg-green-50 rounded border border-green-200">
            <div className="font-bold text-green-800 mb-1">通信量</div>
            <div className="text-slate-700">
              O(2N)，与GPU数量无关
            </div>
          </div>
          <div className="p-3 bg-blue-50 rounded border border-blue-200">
            <div className="font-bold text-blue-800 mb-1">带宽利用</div>
            <div className="text-slate-700">
              100% 网络带宽利用率
            </div>
          </div>
          <div className="p-3 bg-purple-50 rounded border border-purple-200">
            <div className="font-bold text-purple-800 mb-1">可扩展性</div>
            <div className="text-slate-700">
              支持数百GPU并行
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-slate-50 rounded border border-slate-200 font-mono text-xs text-slate-700">
          <div>通信步骤 = 2 × (N - 1) = 2 × 3 = 6 步</div>
          <div>每步传输量 = 梯度总大小 / N = M / 4</div>
          <div>总通信量 = 6 × M/4 = 1.5M ≈ 2M （理论最优）</div>
        </div>
      </div>
    </div>
  )
}

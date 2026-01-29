'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

type Phase = 'forward-gather' | 'forward-compute' | 'forward-release' | 'backward-gather' | 'backward-compute' | 'backward-release' | 'optimizer-update'

export default function DeepSpeedOffloadFlow() {
  const [currentPhase, setCurrentPhase] = useState<Phase>('forward-gather')
  const [isPlaying, setIsPlaying] = useState(false)

  const phases: { id: Phase; name: string; description: string; color: string }[] = [
    {
      id: 'forward-gather',
      name: '前向：All-Gather 参数',
      description: '从 CPU 内存收集参数分片到 GPU',
      color: 'blue'
    },
    {
      id: 'forward-compute',
      name: '前向：GPU 计算',
      description: '使用 GPU 上的完整参数计算激活值',
      color: 'green'
    },
    {
      id: 'forward-release',
      name: '前向：释放参数',
      description: '计算完成后立即释放 GPU 上的参数',
      color: 'orange'
    },
    {
      id: 'backward-gather',
      name: '反向：All-Gather 参数',
      description: '再次从 CPU 收集参数用于梯度计算',
      color: 'blue'
    },
    {
      id: 'backward-compute',
      name: '反向：GPU 计算梯度',
      description: '计算梯度并 Reduce-Scatter 到各 GPU',
      color: 'purple'
    },
    {
      id: 'backward-release',
      name: '反向：释放参数',
      description: '梯度计算完成，再次释放参数',
      color: 'orange'
    },
    {
      id: 'optimizer-update',
      name: '优化器：CPU 更新',
      description: '在 CPU 上执行优化器更新（Adam）',
      color: 'red'
    }
  ]

  const currentPhaseIndex = phases.findIndex(p => p.id === currentPhase)
  const currentPhaseData = phases[currentPhaseIndex]

  const handleNext = () => {
    const nextIndex = (currentPhaseIndex + 1) % phases.length
    setCurrentPhase(phases[nextIndex].id)
  }

  const handlePrev = () => {
    const prevIndex = (currentPhaseIndex - 1 + phases.length) % phases.length
    setCurrentPhase(phases[prevIndex].id)
  }

  const handlePlay = () => {
    setIsPlaying(true)
    const interval = setInterval(() => {
      setCurrentPhase(prev => {
        const idx = phases.findIndex(p => p.id === prev)
        return phases[(idx + 1) % phases.length].id
      })
    }, 2000)

    setTimeout(() => {
      clearInterval(interval)
      setIsPlaying(false)
    }, 14000)
  }

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        DeepSpeed Offload 工作流程
      </h3>

      {/* 架构图 */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {/* CPU 内存 */}
        <div className="col-span-1 bg-red-50 dark:bg-red-900/20 rounded-xl p-4 border-2 border-red-300 dark:border-red-700">
          <div className="text-center mb-3">
            <h4 className="font-bold text-red-900 dark:text-red-300 mb-1">CPU 内存</h4>
            <p className="text-xs text-red-700 dark:text-red-400">优化器状态 + 参数分片</p>
          </div>

          <div className="space-y-2">
            {/* 参数分片 */}
            <motion.div
              className="bg-red-500 text-white rounded-lg p-3 text-center"
              animate={{
                opacity: ['forward-gather', 'backward-gather'].includes(currentPhase) ? 0.5 : 1,
                scale: ['forward-gather', 'backward-gather'].includes(currentPhase) ? 0.95 : 1
              }}
              transition={{ duration: 0.5 }}
            >
              <p className="text-sm font-semibold">参数分片 θ/N</p>
              <p className="text-xs">3.5 GB</p>
            </motion.div>

            {/* 优化器状态 */}
            <motion.div
              className="bg-red-600 text-white rounded-lg p-3 text-center"
              animate={{
                scale: currentPhase === 'optimizer-update' ? 1.05 : 1,
                boxShadow: currentPhase === 'optimizer-update' 
                  ? '0 0 20px rgba(239, 68, 68, 0.6)' 
                  : '0 0 0px rgba(239, 68, 68, 0)'
              }}
              transition={{ duration: 0.5 }}
            >
              <p className="text-sm font-semibold">优化器状态</p>
              <p className="text-xs">Momentum + Variance</p>
              <p className="text-xs">7 GB</p>
            </motion.div>
          </div>

          <AnimatePresence>
            {currentPhase === 'optimizer-update' && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="mt-3 p-2 bg-red-600 rounded-lg text-white text-xs text-center"
              >
                ⚙️ AdamW 更新中...
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* 通信箭头 */}
        <div className="col-span-1 flex flex-col justify-center items-center">
          <AnimatePresence mode="wait">
            {['forward-gather', 'backward-gather'].includes(currentPhase) && (
              <motion.div
                key="to-gpu"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="flex flex-col items-center"
              >
                <div className="text-4xl mb-2">→</div>
                <div className="text-xs text-blue-600 dark:text-blue-400 font-semibold">
                  All-Gather
                </div>
                <div className="text-xs text-gray-300">
                  PCIe 传输
                </div>
              </motion.div>
            )}

            {['forward-release', 'backward-release'].includes(currentPhase) && (
              <motion.div
                key="to-cpu"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex flex-col items-center"
              >
                <div className="text-4xl mb-2">←</div>
                <div className="text-xs text-orange-600 dark:text-orange-400 font-semibold">
                  释放参数
                </div>
                <div className="text-xs text-gray-300">
                  节省 GPU 显存
                </div>
              </motion.div>
            )}

            {['forward-compute', 'backward-compute'].includes(currentPhase) && (
              <motion.div
                key="computing"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex flex-col items-center"
              >
                <div className="text-4xl mb-2">⚡</div>
                <div className="text-xs text-green-600 dark:text-green-400 font-semibold">
                  GPU 计算中
                </div>
              </motion.div>
            )}

            {currentPhase === 'optimizer-update' && (
              <motion.div
                key="cpu-update"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex flex-col items-center"
              >
                <div className="text-4xl mb-2">🔄</div>
                <div className="text-xs text-red-600 dark:text-red-400 font-semibold">
                  CPU 优化器
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* GPU 显存 */}
        <div className="col-span-1 bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border-2 border-green-300 dark:border-green-700">
          <div className="text-center mb-3">
            <h4 className="font-bold text-green-900 dark:text-green-300 mb-1">GPU 显存</h4>
            <p className="text-xs text-green-700 dark:text-green-400">临时加载参数 + 梯度分片</p>
          </div>

          <div className="space-y-2">
            {/* 临时参数（仅计算时存在） */}
            <AnimatePresence>
              {['forward-gather', 'forward-compute', 'backward-gather', 'backward-compute'].includes(currentPhase) && (
                <motion.div
                  key="temp-params"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-blue-500 text-white rounded-lg p-3 text-center"
                >
                  <p className="text-sm font-semibold">
                    {['forward-gather', 'forward-compute'].includes(currentPhase) ? '前向参数' : '反向参数'}
                  </p>
                  <p className="text-xs">临时加载：14 GB</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* 激活值 */}
            <motion.div
              className="bg-green-500 text-white rounded-lg p-3 text-center"
              animate={{
                scale: currentPhase === 'forward-compute' ? 1.05 : 1
              }}
            >
              <p className="text-sm font-semibold">激活值</p>
              <p className="text-xs">20 GB</p>
            </motion.div>

            {/* 梯度分片 */}
            <motion.div
              className="bg-purple-500 text-white rounded-lg p-3 text-center"
              animate={{
                scale: currentPhase === 'backward-compute' ? 1.05 : 1,
                boxShadow: currentPhase === 'backward-compute'
                  ? '0 0 20px rgba(168, 85, 247, 0.6)'
                  : '0 0 0px rgba(168, 85, 247, 0)'
              }}
            >
              <p className="text-sm font-semibold">梯度分片 ∇L/N</p>
              <p className="text-xs">3.5 GB</p>
            </motion.div>
          </div>

          <div className="mt-3 p-2 bg-green-600 rounded-lg text-white text-xs text-center">
            总计：{['forward-gather', 'forward-compute', 'backward-gather', 'backward-compute'].includes(currentPhase) 
              ? '~37.5 GB' 
              : '~23.5 GB'
            }
          </div>
        </div>
      </div>

      {/* 当前阶段说明 */}
      <div className={`p-6 rounded-xl bg-${currentPhaseData.color}-50 dark:bg-${currentPhaseData.color}-900/20 border-2 border-${currentPhaseData.color}-300 dark:border-${currentPhaseData.color}-700 mb-6`}>
        <h4 className={`text-lg font-bold text-${currentPhaseData.color}-900 dark:text-${currentPhaseData.color}-300 mb-2`}>
          阶段 {currentPhaseIndex + 1}/7: {currentPhaseData.name}
        </h4>
        <p className={`text-${currentPhaseData.color}-800 dark:text-${currentPhaseData.color}-200`}>
          {currentPhaseData.description}
        </p>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-3">
        <button
          onClick={handlePrev}
          disabled={isPlaying}
          className="px-6 py-2 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 text-white rounded-lg transition-colors"
        >
          上一步
        </button>
        <button
          onClick={handlePlay}
          disabled={isPlaying}
          className="px-6 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white rounded-lg transition-colors"
        >
          {isPlaying ? '播放中...' : '自动播放'}
        </button>
        <button
          onClick={handleNext}
          disabled={isPlaying}
          className="px-6 py-2 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 text-white rounded-lg transition-colors"
        >
          下一步
        </button>
      </div>

      {/* 性能权衡 */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">✅ 优势</h5>
          <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
            <li>• 显存占用降低 50%-70%</li>
            <li>• 支持更大模型训练</li>
            <li>• 无需昂贵的高显存 GPU</li>
          </ul>
        </div>

        <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
          <h5 className="font-semibold text-orange-900 dark:text-orange-300 mb-2">⚠️ 代价</h5>
          <ul className="text-sm text-orange-800 dark:text-orange-200 space-y-1">
            <li>• 训练速度下降 20%-50%</li>
            <li>• PCIe 带宽成为瓶颈</li>
            <li>• CPU 内存需求增加</li>
          </ul>
        </div>
      </div>

      {/* 通信开销分析 */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-3">📊 通信开销分析</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
          <div>
            <p className="font-semibold mb-1">GPU ↔ CPU 传输（每步）：</p>
            <p>• 前向 All-Gather: |θ| = 14 GB</p>
            <p>• 反向 All-Gather: |θ| = 14 GB</p>
            <p className="mt-1 font-bold">总计：28 GB/step</p>
          </div>
          <div>
            <p className="font-semibold mb-1">PCIe 4.0 x16 理论带宽：</p>
            <p>• 32 GB/s（双向）</p>
            <p>• 传输时间：28 GB ÷ 32 GB/s ≈ 0.875s</p>
            <p className="mt-1 font-bold text-orange-600 dark:text-orange-400">占总训练时间 30%-40%</p>
          </div>
        </div>
      </div>
    </div>
  )
}

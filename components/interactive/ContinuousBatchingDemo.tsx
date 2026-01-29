'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Request {
  id: number
  arrivalTime: number
  generationLength: number
  currentToken: number
  color: string
  status: 'waiting' | 'generating' | 'done'
}

export default function ContinuousBatchingDemo() {
  const [mode, setMode] = useState<'static' | 'continuous'>('static')
  const [isRunning, setIsRunning] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [requests, setRequests] = useState<Request[]>([
    { id: 1, arrivalTime: 0, generationLength: 8, currentToken: 0, color: 'blue', status: 'waiting' },
    { id: 2, arrivalTime: 0, generationLength: 12, currentToken: 0, color: 'green', status: 'waiting' },
    { id: 3, arrivalTime: 0, generationLength: 6, currentToken: 0, color: 'purple', status: 'waiting' },
    { id: 4, arrivalTime: 5, generationLength: 10, currentToken: 0, color: 'orange', status: 'waiting' },
  ])

  const maxBatchSize = 4
  const generationSpeed = 1 // 1 token per step

  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      setCurrentTime((t) => t + 1)
      setRequests((reqs) => {
        const newReqs = [...reqs]

        if (mode === 'static') {
          // Static Batching: 处理前 maxBatchSize 个未完成的请求
          const activeRequests = newReqs.filter(
            (r) => r.status !== 'done' && r.arrivalTime <= currentTime
          ).slice(0, maxBatchSize)

          const allDone = activeRequests.every((r) => r.currentToken >= r.generationLength)

          if (allDone && activeRequests.length > 0) {
            // 所有请求完成，标记为 done
            activeRequests.forEach((r) => {
              r.status = 'done'
            })
          } else {
            // 继续生成
            activeRequests.forEach((r) => {
              if (r.currentToken < r.generationLength) {
                r.status = 'generating'
                r.currentToken += generationSpeed
              }
            })
          }

          // 等待中的请求
          newReqs.forEach((r) => {
            if (r.status === 'waiting' && !activeRequests.includes(r) && r.arrivalTime <= currentTime) {
              // 保持 waiting 状态
            }
          })
        } else {
          // Continuous Batching: 动态添加/移除
          const activeRequests = newReqs.filter(
            (r) => r.status !== 'done' && r.arrivalTime <= currentTime
          ).slice(0, maxBatchSize)

          activeRequests.forEach((r) => {
            if (r.currentToken < r.generationLength) {
              r.status = 'generating'
              r.currentToken += generationSpeed
            } else {
              r.status = 'done'
            }
          })
        }

        return newReqs
      })
    }, 500)

    return () => clearInterval(interval)
  }, [isRunning, currentTime, mode])

  const reset = () => {
    setIsRunning(false)
    setCurrentTime(0)
    setRequests([
      { id: 1, arrivalTime: 0, generationLength: 8, currentToken: 0, color: 'blue', status: 'waiting' },
      { id: 2, arrivalTime: 0, generationLength: 12, currentToken: 0, color: 'green', status: 'waiting' },
      { id: 3, arrivalTime: 0, generationLength: 6, currentToken: 0, color: 'purple', status: 'waiting' },
      { id: 4, arrivalTime: 5, generationLength: 10, currentToken: 0, color: 'orange', status: 'waiting' },
    ])
  }

  const activeRequests = requests.filter((r) => r.status === 'generating').length
  const completedRequests = requests.filter((r) => r.status === 'done').length
  const totalTokens = requests.reduce((sum, r) => sum + r.currentToken, 0)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        Continuous Batching vs Static Batching
      </h3>

      {/* 控制面板 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="flex gap-2">
          <button
            onClick={() => { setMode('static'); reset(); }}
            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-all ${
              mode === 'static'
                ? 'bg-red-500 text-white scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-100'
            }`}
          >
            Static Batching
          </button>
          <button
            onClick={() => { setMode('continuous'); reset(); }}
            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-all ${
              mode === 'continuous'
                ? 'bg-green-500 text-white scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-100'
            }`}
          >
            Continuous Batching
          </button>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600"
          >
            {isRunning ? '⏸ 暂停' : '▶️ 开始'}
          </button>
          <button
            onClick={reset}
            className="flex-1 px-4 py-2 bg-gray-500 text-white rounded-lg font-semibold hover:bg-gray-600"
          >
            🔄 重置
          </button>
        </div>

        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-sm text-blue-700 dark:text-blue-400">当前时刻</p>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-200">
            {currentTime}
          </p>
        </div>
      </div>

      {/* 请求可视化 */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-xl">
        <h4 className={`text-lg font-bold mb-4 ${
          mode === 'static' ? 'text-red-700 dark:text-red-400' : 'text-green-700 dark:text-green-400'
        }`}>
          {mode === 'static' ? '⚠️ Static Batching：等待最慢请求' : '✅ Continuous Batching：动态调度'}
        </h4>

        <div className="space-y-3">
          {requests.map((req) => {
            const isVisible = req.arrivalTime <= currentTime
            const isActive = req.status === 'generating'
            const isDone = req.status === 'done'

            return (
              <AnimatePresence key={req.id}>
                {isVisible && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`p-4 rounded-lg border-2 ${
                      isDone
                        ? 'border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800'
                        : isActive
                        ? `border-${req.color}-500 bg-${req.color}-50 dark:bg-${req.color}-900/20`
                        : 'border-gray-400 dark:border-gray-500 bg-white dark:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center gap-4 mb-2">
                      <span className={`font-bold ${
                        isDone ? 'text-gray-500' : `text-${req.color}-700 dark:text-${req.color}-300`
                      }`}>
                        Request {req.id}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        isDone
                          ? 'bg-gray-400 text-white'
                          : isActive
                          ? `bg-${req.color}-500 text-white`
                          : 'bg-yellow-400 text-yellow-900'
                      }`}>
                        {isDone ? 'Done' : isActive ? 'Generating' : 'Waiting'}
                      </span>
                      <span className="text-sm text-gray-300">
                        到达时刻: t={req.arrivalTime}
                      </span>
                    </div>

                    <div className="relative h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        className={`h-full ${isDone ? 'bg-gray-400' : `bg-${req.color}-500`}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${(req.currentToken / req.generationLength) * 100}%` }}
                        transition={{ duration: 0.3 }}
                      />
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-gray-100">
                        {req.currentToken} / {req.generationLength} tokens
                      </span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            )
          })}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-sm text-blue-700 dark:text-blue-400 mb-1">正在生成</p>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-200">
            {activeRequests}
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">/ {maxBatchSize} 最大</p>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-sm text-green-700 dark:text-green-400 mb-1">已完成</p>
          <p className="text-2xl font-bold text-green-900 dark:text-green-200">
            {completedRequests}
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">/ {requests.length} 总数</p>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">总生成量</p>
          <p className="text-2xl font-bold text-purple-900 dark:text-purple-200">
            {totalTokens}
          </p>
          <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">tokens</p>
        </div>
      </div>

      {/* 对比说明 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <h5 className="font-semibold text-red-900 dark:text-red-300 mb-2">
            ⚠️ Static Batching 问题
          </h5>
          <ul className="text-sm text-red-800 dark:text-red-200 space-y-1">
            <li>• 必须等待整个 batch 最慢的请求完成</li>
            <li>• 短请求完成后仍占用 batch slot</li>
            <li>• 新请求需等待当前 batch 全部结束</li>
            <li>• GPU 利用率低（等待时间长）</li>
          </ul>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            ✅ Continuous Batching 优势
          </h5>
          <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
            <li>• 请求完成后立即释放 slot</li>
            <li>• 新请求立即插入可用 slot</li>
            <li>• GPU 利用率高（持续满载）</li>
            <li>• 吞吐量提升 <strong>2-3x</strong></li>
          </ul>
        </div>
      </div>

      {/* 性能公式 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">吞吐量对比：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`Static Batching:
  Throughput = Σ(L_i) / max(L_1, L_2, ..., L_N)
  
Continuous Batching:
  Throughput = Σ(L_i) / avg(完成时间)

实测加速比：2.14x（vLLM LLaMA-13B）`}
        </pre>
      </div>
    </div>
  )
}

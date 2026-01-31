'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Request {
  id: string
  text: string
  timestamp: number
  status: 'queued' | 'batching' | 'processing' | 'completed'
  batchId?: number
}

const exampleTexts = [
  "This movie is amazing!",
  "Terrible experience",
  "Absolutely loved it",
  "Not recommended",
  "Best film ever",
  "Waste of time",
  "Highly recommend",
  "Disappointing",
]

export default function RequestQueueVisualizer() {
  const [mode, setMode] = useState<'sync' | 'batch'>('sync')
  const [requests, setRequests] = useState<Request[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [batchSize, setBatchSize] = useState(4)
  const [stats, setStats] = useState({
    totalProcessed: 0,
    avgLatency: 0,
    throughput: 0,
  })

  // 添加新请求
  const addRequest = useCallback(() => {
    const newRequest: Request = {
      id: Math.random().toString(36).substr(2, 9),
      text: exampleTexts[Math.floor(Math.random() * exampleTexts.length)],
      timestamp: Date.now(),
      status: 'queued',
    }
    setRequests(prev => [...prev, newRequest])
  }, [])

  // 自动添加请求
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      addRequest()
    }, mode === 'sync' ? 500 : 200)

    return () => clearInterval(interval)
  }, [isRunning, mode, addRequest])

  // 处理请求
  useEffect(() => {
    if (mode === 'sync') {
      // 同步模式：逐个处理
      const queuedRequests = requests.filter(r => r.status === 'queued')
      if (queuedRequests.length > 0 && !requests.some(r => r.status === 'processing')) {
        const nextRequest = queuedRequests[0]

        // 开始处理
        setRequests(prev =>
          prev.map(r => r.id === nextRequest.id ? { ...r, status: 'processing' as const } : r)
        )

        // 模拟处理时间
        setTimeout(() => {
          setRequests(prev => {
            const updated = prev.map(r =>
              r.id === nextRequest.id ? { ...r, status: 'completed' as const } : r
            )

            // 计算统计
            const latency = Date.now() - nextRequest.timestamp
            setStats(s => ({
              totalProcessed: s.totalProcessed + 1,
              avgLatency: (s.avgLatency * s.totalProcessed + latency) / (s.totalProcessed + 1),
              throughput: s.totalProcessed / ((Date.now() - (requests[0]?.timestamp || Date.now())) / 1000) || 0,
            }))

            return updated
          })

          // 移除已完成的请求
          setTimeout(() => {
            setRequests(prev => prev.filter(r => r.status !== 'completed'))
          }, 500)
        }, 800)
      }
    } else {
      // 批处理模式：批量处理
      const queuedRequests = requests.filter(r => r.status === 'queued')

      if (queuedRequests.length >= batchSize && !requests.some(r => r.status === 'batching' || r.status === 'processing')) {
        const batch = queuedRequests.slice(0, batchSize)
        const batchId = Date.now()

        // 标记为批处理中
        setRequests(prev =>
          prev.map(r =>
            batch.some(b => b.id === r.id)
              ? { ...r, status: 'batching' as const, batchId }
              : r
          )
        )

        // 等待批次形成
        setTimeout(() => {
          setRequests(prev =>
            prev.map(r => r.batchId === batchId ? { ...r, status: 'processing' as const } : r)
          )

          // 批量处理
          setTimeout(() => {
            setRequests(prev => {
              const updated = prev.map(r =>
                r.batchId === batchId ? { ...r, status: 'completed' as const } : r
              )

              // 计算统计
              batch.forEach(req => {
                const latency = Date.now() - req.timestamp
                setStats(s => ({
                  totalProcessed: s.totalProcessed + 1,
                  avgLatency: (s.avgLatency * s.totalProcessed + latency) / (s.totalProcessed + 1),
                  throughput: s.totalProcessed / ((Date.now() - (requests[0]?.timestamp || Date.now())) / 1000) || 0,
                }))
              })

              return updated
            })

            // 移除已完成的请求
            setTimeout(() => {
              setRequests(prev => prev.filter(r => r.status !== 'completed'))
            }, 500)
          }, 600)
        }, 300)
      }
    }
  }, [requests, mode, batchSize])

  const reset = () => {
    setRequests([])
    setStats({ totalProcessed: 0, avgLatency: 0, throughput: 0 })
    setIsRunning(false)
  }

  const queuedCount = requests.filter(r => r.status === 'queued').length
  const processingCount = requests.filter(r => r.status === 'processing' || r.status === 'batching').length

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          📊 请求队列与批处理可视化
        </h3>
        <p className="text-slate-600">
          对比同步处理与批处理模式的性能差异
        </p>
      </div>

      {/* 模式选择 */}
      <div className="flex gap-3 mb-6 justify-center">
        <button
          onClick={() => { setMode('sync'); reset(); }}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${mode === 'sync'
            ? 'bg-blue-600 text-white shadow-lg scale-105'
            : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
        >
          🔄 同步处理
        </button>
        <button
          onClick={() => { setMode('batch'); reset(); }}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${mode === 'batch'
            ? 'bg-green-600 text-white shadow-lg scale-105'
            : 'bg-white text-slate-700 hover:bg-slate-100'
            }`}
        >
          📦 批处理模式
        </button>
      </div>

      {/* 批大小控制 (仅批处理模式) */}
      {mode === 'batch' && (
        <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            批大小：{batchSize}
          </label>
          <input
            type="range"
            min="2"
            max="8"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="w-full"
            disabled={isRunning}
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>2</span>
            <span>4</span>
            <span>6</span>
            <span>8</span>
          </div>
        </div>
      )}

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6 justify-center">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`px-6 py-2 rounded-lg font-medium text-white ${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
            }`}
        >
          {isRunning ? '⏸️ 暂停' : '▶️ 开始'}
        </button>
        <button
          onClick={addRequest}
          disabled={isRunning}
          className="px-6 py-2 rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:bg-slate-300"
        >
          ➕ 手动添加请求
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 rounded-lg font-medium bg-slate-600 text-white hover:bg-slate-700"
        >
          🔄 重置
        </button>
      </div>

      {/* 统计信息 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg border border-slate-200 text-center">
          <div className="text-2xl font-bold text-blue-600">{queuedCount}</div>
          <div className="text-sm text-slate-600">队列中</div>
        </div>
        <div className="bg-white p-4 rounded-lg border border-slate-200 text-center">
          <div className="text-2xl font-bold text-green-600">{stats.totalProcessed}</div>
          <div className="text-sm text-slate-600">已处理</div>
        </div>
        <div className="bg-white p-4 rounded-lg border border-slate-200 text-center">
          <div className="text-2xl font-bold text-purple-600">
            {stats.throughput.toFixed(1)} RPS
          </div>
          <div className="text-sm text-slate-600">吞吐量</div>
        </div>
      </div>

      {/* 可视化区域 */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 min-h-[400px]">
        <div className="flex items-start gap-8">
          {/* 请求队列 */}
          <div className="flex-1">
            <div className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
              请求队列
            </div>
            <div className="space-y-2 max-h-[350px] overflow-y-auto">
              <AnimatePresence>
                {requests.filter(r => r.status === 'queued').map((req) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg"
                  >
                    <div className="text-xs text-slate-600 truncate">
                      {req.text}
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      ID: {req.id.substr(0, 6)}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* 处理区域 */}
          <div className="flex-1">
            <div className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-500"></span>
              {mode === 'batch' ? '批处理中' : '处理中'}
            </div>
            <div className="space-y-2">
              <AnimatePresence>
                {requests.filter(r => r.status === 'batching').map((req) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="p-3 bg-orange-50 border border-orange-300 rounded-lg"
                  >
                    <div className="text-xs text-slate-600 truncate">
                      {req.text}
                    </div>
                    <div className="text-xs text-orange-600 mt-1 font-medium">
                      🔄 等待批次形成...
                    </div>
                  </motion.div>
                ))}
                {requests.filter(r => r.status === 'processing').map((req) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="p-3 bg-blue-50 border border-blue-300 rounded-lg relative overflow-hidden"
                  >
                    <motion.div
                      className="absolute inset-0 bg-blue-200 opacity-30"
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: mode === 'batch' ? 0.6 : 0.8 }}
                    />
                    <div className="relative z-10">
                      <div className="text-xs text-slate-600 truncate">
                        {req.text}
                      </div>
                      <div className="text-xs text-blue-600 mt-1 font-medium">
                        ⚙️ 推理中...
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* 完成区域 */}
          <div className="flex-1">
            <div className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-green-500"></span>
              已完成
            </div>
            <div className="space-y-2">
              <AnimatePresence>
                {requests.filter(r => r.status === 'completed').map((req) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="p-3 bg-green-50 border border-green-200 rounded-lg"
                  >
                    <div className="text-xs text-slate-600 truncate">
                      {req.text}
                    </div>
                    <div className="text-xs text-green-600 mt-1 font-medium">
                      ✅ 完成
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>

      {/* 性能对比 */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-sm font-medium text-slate-700 mb-2">
            平均延迟
          </div>
          <div className="text-3xl font-bold text-blue-600">
            {stats.avgLatency.toFixed(0)} ms
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {mode === 'sync' ? '同步模式通常 800-1000ms' : '批处理增加等待时间，但提升吞吐量'}
          </div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="text-sm font-medium text-slate-700 mb-2">
            理论提升
          </div>
          <div className="text-3xl font-bold text-green-600">
            {mode === 'batch' ? `${batchSize}x` : '1x'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {mode === 'batch'
              ? `批大小 ${batchSize}，GPU 并行处理`
              : '单个请求串行处理'}
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm font-medium text-blue-900 mb-2">
          💡 关键洞察
        </div>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>同步模式</strong>：逐个处理请求，延迟低但吞吐量有限</li>
          <li>• <strong>批处理模式</strong>：累积请求批量处理，GPU 利用率更高</li>
          <li>• <strong>权衡</strong>：批处理增加等待时间，但显著提升总吞吐量（3-5x）</li>
          <li>• <strong>生产实践</strong>：通常设置 max_wait_time=50ms，batch_size=8-16</li>
        </ul>
      </div>
    </div>
  )
}

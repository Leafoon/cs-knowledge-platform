'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, AlertTriangle, Clock, TrendingUp, RefreshCw, User, Activity } from 'lucide-react'

interface RequestLog {
  id: number
  timestamp: string
  userId: string
  tier: string
  allowed: boolean
}

export default function RateLimitingVisualizer() {
  const [selectedTier, setSelectedTier] = useState<'free' | 'basic' | 'premium'>('basic')
  const [requestCount, setRequestCount] = useState(0)
  const [requestLogs, setRequestLogs] = useState<RequestLog[]>([])
  const [isThrottled, setIsThrottled] = useState(false)
  const [timeWindow, setTimeWindow] = useState(60)
  
  const tiers = {
    free: { limit: 5, color: 'gray', name: '免费版', rpm: 5 },
    basic: { limit: 20, color: 'blue', name: '基础版', rpm: 20 },
    premium: { limit: 100, color: 'purple', name: '高级版', rpm: 100 },
  }
  
  const currentTier = tiers[selectedTier]
  const utilizationPercent = Math.min((requestCount / currentTier.limit) * 100, 100)
  
  // 自动衰减请求计数（模拟时间窗口滑动）
  useEffect(() => {
    const interval = setInterval(() => {
      setRequestCount(prev => Math.max(0, prev - 1))
      
      // 清理旧日志
      setRequestLogs(prev => {
        const now = Date.now()
        return prev.filter(log => now - new Date(log.timestamp).getTime() < timeWindow * 1000)
      })
      
      // 检查是否仍在限流
      if (requestCount < currentTier.limit) {
        setIsThrottled(false)
      }
    }, timeWindow * 1000 / currentTier.limit)
    
    return () => clearInterval(interval)
  }, [requestCount, currentTier.limit, timeWindow])
  
  const sendRequest = () => {
    const now = new Date()
    const log: RequestLog = {
      id: Date.now(),
      timestamp: now.toISOString(),
      userId: `user-${selectedTier}`,
      tier: selectedTier,
      allowed: requestCount < currentTier.limit
    }
    
    if (requestCount < currentTier.limit) {
      setRequestCount(prev => prev + 1)
      setRequestLogs(prev => [log, ...prev].slice(0, 10))
    } else {
      setIsThrottled(true)
      setRequestLogs(prev => [log, ...prev].slice(0, 10))
    }
  }
  
  const resetDemo = () => {
    setRequestCount(0)
    setRequestLogs([])
    setIsThrottled(false)
  }
  
  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Zap className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-gray-800">速率限制可视化演示</h3>
        </div>
        <p className="text-gray-600">观察令牌桶算法如何控制请求频率</p>
      </div>

      {/* 用户等级选择 */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {(Object.keys(tiers) as Array<keyof typeof tiers>).map((tier) => {
          const config = tiers[tier]
          const isSelected = selectedTier === tier
          
          return (
            <motion.button
              key={tier}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                setSelectedTier(tier)
                resetDemo()
              }}
              className={`p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? `bg-${config.color}-50 border-${config.color}-500 shadow-md`
                  : 'bg-white border-gray-200 hover:border-gray-300'
              }`}
            >
              <User className={`w-6 h-6 mx-auto mb-2 ${
                isSelected ? `text-${config.color}-600` : 'text-gray-400'
              }`} />
              <h4 className={`font-semibold mb-1 ${
                isSelected ? `text-${config.color}-700` : 'text-gray-700'
              }`}>
                {config.name}
              </h4>
              <p className="text-sm text-gray-600">{config.rpm} 请求/分钟</p>
            </motion.button>
          )
        })}
      </div>

      {/* 令牌桶可视化 */}
      <div className="bg-white rounded-lg p-6 shadow-inner mb-6">
        <div className="flex items-center justify-between mb-6">
          <h4 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-600" />
            令牌桶状态
          </h4>
          <div className="text-sm text-gray-600">
            <Clock className="w-4 h-4 inline mr-1" />
            时间窗口: {timeWindow}秒
          </div>
        </div>

        {/* 进度条 */}
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">
              已使用: {requestCount} / {currentTier.limit}
            </span>
            <span className={`text-sm font-semibold ${
              utilizationPercent > 80 ? 'text-red-600' :
              utilizationPercent > 50 ? 'text-yellow-600' :
              'text-green-600'
            }`}>
              {utilizationPercent.toFixed(0)}%
            </span>
          </div>
          
          <div className="h-8 bg-gray-200 rounded-lg overflow-hidden relative">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${utilizationPercent}%` }}
              transition={{ duration: 0.3 }}
              className={`h-full ${
                utilizationPercent > 80 ? 'bg-gradient-to-r from-red-400 to-red-600' :
                utilizationPercent > 50 ? 'bg-gradient-to-r from-yellow-400 to-yellow-600' :
                'bg-gradient-to-r from-green-400 to-green-600'
              }`}
            />
            
            {/* 限流警告线 */}
            <div 
              className="absolute top-0 bottom-0 w-0.5 bg-red-500"
              style={{ left: '80%' }}
            >
              <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-red-500 rounded-full" />
            </div>
          </div>
        </div>

        {/* 令牌可视化 */}
        <div className="grid grid-cols-10 gap-2 mb-6">
          {Array.from({ length: Math.min(currentTier.limit, 50) }).map((_, idx) => (
            <motion.div
              key={idx}
              initial={{ scale: 0 }}
              animate={{ 
                scale: idx < requestCount ? 0.7 : 1,
                opacity: idx < requestCount ? 0.3 : 1
              }}
              transition={{ delay: idx * 0.01 }}
              className={`aspect-square rounded-full ${
                idx < requestCount
                  ? 'bg-gray-300'
                  : utilizationPercent > 80
                  ? 'bg-red-400'
                  : 'bg-green-400'
              } shadow-sm`}
            />
          ))}
        </div>

        {/* 操作按钮 */}
        <div className="flex gap-3">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={sendRequest}
            disabled={isThrottled}
            className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
              isThrottled
                ? 'bg-red-100 text-red-600 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isThrottled ? '⛔ 已被限流' : '📤 发送请求'}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={resetDemo}
            className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
          </motion.button>
        </div>

        {/* 限流警告 */}
        <AnimatePresence>
          {isThrottled && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                <div>
                  <h5 className="font-semibold text-red-900 mb-1">速率限制触发</h5>
                  <p className="text-sm text-red-800">
                    您在 {timeWindow} 秒内的请求次数已达到 {currentTier.limit} 次上限。
                    请等待令牌补充后再试。
                  </p>
                  <p className="text-xs text-red-700 mt-2">
                    HTTP 429 Too Many Requests · Retry-After: {timeWindow}s
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 请求日志 */}
      <div className="bg-white rounded-lg p-6 shadow-inner">
        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-indigo-600" />
          请求日志
        </h4>
        
        <div className="space-y-2 max-h-64 overflow-y-auto">
          <AnimatePresence>
            {requestLogs.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                暂无请求记录
              </div>
            ) : (
              requestLogs.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className={`p-3 rounded-lg border ${
                    log.allowed
                      ? 'bg-green-50 border-green-200'
                      : 'bg-red-50 border-red-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        log.allowed ? 'bg-green-500' : 'bg-red-500'
                      }`} />
                      <span className="text-sm font-mono text-gray-700">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-600">
                        {log.userId}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded font-medium ${
                        log.tier === 'premium' ? 'bg-purple-100 text-purple-700' :
                        log.tier === 'basic' ? 'bg-blue-100 text-blue-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {tiers[log.tier as keyof typeof tiers].name}
                      </span>
                    </div>
                    
                    <span className={`text-sm font-semibold ${
                      log.allowed ? 'text-green-700' : 'text-red-700'
                    }`}>
                      {log.allowed ? '✓ 200 OK' : '✗ 429 Rate Limited'}
                    </span>
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* 算法说明 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mt-6 p-4 bg-indigo-50 border-l-4 border-indigo-500 rounded-r-lg"
      >
        <h5 className="font-semibold text-indigo-900 mb-2">令牌桶算法原理</h5>
        <div className="text-sm text-indigo-800 space-y-1">
          <p>1. 桶以固定速率补充令牌（如每秒补充 N 个）</p>
          <p>2. 每个请求消耗一个令牌</p>
          <p>3. 令牌不足时请求被拒绝（返回 429 错误）</p>
          <p>4. 允许短时间的突发流量（桶内有余量时）</p>
        </div>
      </motion.div>
    </div>
  )
}

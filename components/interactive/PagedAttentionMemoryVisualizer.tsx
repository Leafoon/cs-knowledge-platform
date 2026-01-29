'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

export default function PagedAttentionMemoryVisualizer() {
  const [activeView, setActiveView] = useState<'traditional' | 'paged'>('traditional')
  const [numRequests, setNumRequests] = useState(3)

  // 模拟请求数据
  const requests = [
    { id: 1, actualLen: 42, color: 'blue' },
    { id: 2, actualLen: 128, color: 'green' },
    { id: 3, actualLen: 67, color: 'purple' },
    { id: 4, actualLen: 95, color: 'orange' },
  ].slice(0, numRequests)

  const maxSeqLen = 256
  const blockSize = 16

  // 计算传统方法的显存
  const traditionalMemory = requests.length * maxSeqLen
  const traditionalWaste = requests.reduce((sum, req) => sum + (maxSeqLen - req.actualLen), 0)

  // 计算 Paged Attention 的显存
  const pagedMemory = requests.reduce((sum, req) => {
    const blocksNeeded = Math.ceil(req.actualLen / blockSize)
    return sum + blocksNeeded * blockSize
  }, 0)

  const wastePercentage = (traditionalWaste / traditionalMemory * 100).toFixed(1)
  const savings = ((traditionalMemory - pagedMemory) / traditionalMemory * 100).toFixed(1)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        PagedAttention 内存分配可视化
      </h3>

      {/* 控制面板 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-2">
            并发请求数：{numRequests}
          </label>
          <input
            type="range"
            min="1"
            max="4"
            value={numRequests}
            onChange={(e) => setNumRequests(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setActiveView('traditional')}
            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-all ${
              activeView === 'traditional'
                ? 'bg-red-500 text-white scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-100'
            }`}
          >
            传统 KV Cache
          </button>
          <button
            onClick={() => setActiveView('paged')}
            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-all ${
              activeView === 'paged'
                ? 'bg-green-500 text-white scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-100'
            }`}
          >
            PagedAttention
          </button>
        </div>
      </div>

      {/* 可视化区域 */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-xl">
        {activeView === 'traditional' ? (
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-red-700 dark:text-red-400 mb-3">
              ⚠️ 传统 KV Cache：连续分配
            </h4>
            {requests.map((req, idx) => (
              <motion.div
                key={req.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="space-y-2"
              >
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-gray-100 w-24">
                    Request {req.id}
                  </span>
                  <div className="flex-1 relative h-8 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                    {/* 实际使用部分 */}
                    <div
                      className={`absolute h-full bg-${req.color}-500`}
                      style={{ width: `${(req.actualLen / maxSeqLen) * 100}%` }}
                    >
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">
                        {req.actualLen} tokens
                      </span>
                    </div>
                    {/* 浪费部分 */}
                    <div
                      className="absolute h-full bg-red-300 dark:bg-red-800 opacity-50"
                      style={{
                        left: `${(req.actualLen / maxSeqLen) * 100}%`,
                        width: `${((maxSeqLen - req.actualLen) / maxSeqLen) * 100}%`,
                      }}
                    >
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-red-900 dark:text-red-200">
                        浪费 {maxSeqLen - req.actualLen}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs text-gray-300 w-32">
                    预分配：{maxSeqLen}
                  </span>
                </div>
              </motion.div>
            ))}
            <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 rounded-lg text-center">
              <p className="text-sm text-red-800 dark:text-red-300">
                总显存：<strong>{traditionalMemory} tokens</strong>，浪费：<strong>{traditionalWaste} tokens ({wastePercentage}%)</strong>
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-green-700 dark:text-green-400 mb-3">
              ✅ PagedAttention：按需分配（块大小={blockSize}）
            </h4>
            {requests.map((req, idx) => {
              const blocksNeeded = Math.ceil(req.actualLen / blockSize)
              const allocatedTokens = blocksNeeded * blockSize
              const internalWaste = allocatedTokens - req.actualLen

              return (
                <motion.div
                  key={req.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="space-y-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-gray-100 w-24">
                      Request {req.id}
                    </span>
                    <div className="flex-1 flex gap-1">
                      {Array.from({ length: blocksNeeded }, (_, i) => {
                        const tokensInBlock = Math.min(blockSize, req.actualLen - i * blockSize)
                        const isFull = tokensInBlock === blockSize

                        return (
                          <div
                            key={i}
                            className={`relative h-8 flex-1 rounded ${
                              isFull
                                ? `bg-${req.color}-500`
                                : `bg-${req.color}-400`
                            } flex items-center justify-center`}
                          >
                            <span className="text-xs font-bold text-white">
                              Block {i}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                    <span className="text-xs text-gray-300 w-32">
                      {blocksNeeded} blocks
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-300 ml-24">
                    <span>实际：{req.actualLen} tokens</span>
                    <span>分配：{allocatedTokens} tokens</span>
                    {internalWaste > 0 && (
                      <span className="text-yellow-600 dark:text-yellow-400">
                        块内浪费：{internalWaste}
                      </span>
                    )}
                  </div>
                </motion.div>
              )
            })}
            <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded-lg text-center">
              <p className="text-sm text-green-800 dark:text-green-300">
                总显存：<strong>{pagedMemory} tokens</strong>，节省：<strong>{traditionalMemory - pagedMemory} tokens ({savings}%)</strong>
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 对比表 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-sm text-blue-700 dark:text-blue-400 mb-1">传统方法显存</p>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-200">
            {traditionalMemory}
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">tokens</p>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-sm text-green-700 dark:text-green-400 mb-1">PagedAttention 显存</p>
          <p className="text-2xl font-bold text-green-900 dark:text-green-200">
            {pagedMemory}
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">tokens</p>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">节省比例</p>
          <p className="text-2xl font-bold text-purple-900 dark:text-purple-200">
            {savings}%
          </p>
          <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">更高效</p>
        </div>
      </div>

      {/* 核心优势 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            ✅ 按需分配
          </h5>
          <p className="text-sm text-green-800 dark:text-green-200">
            仅分配实际需要的块数，避免预分配浪费
          </p>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            ✅ 内存共享
          </h5>
          <p className="text-sm text-green-800 dark:text-green-200">
            Copy-on-Write：共享相同 Prompt 的块
          </p>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            ✅ 无碎片化
          </h5>
          <p className="text-sm text-green-800 dark:text-green-200">
            固定块大小，物理内存高效管理
          </p>
        </div>
      </div>

      {/* 公式说明 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">显存节省公式：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`传统显存 = N × max_seq_len × D_kv
PagedAttention 显存 = Σ(⌈actual_len_i / block_size⌉ × block_size) × D_kv

节省比例 = 1 - (PagedAttention / 传统)
实测：60-80% 节省（高并发场景）`}
        </pre>
      </div>
    </div>
  )
}

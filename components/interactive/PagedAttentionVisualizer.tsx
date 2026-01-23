'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Database, Box, Share2 } from 'lucide-react'

interface Block {
  id: number
  tokens: string[]
  owner: number | null
  shared: boolean
}

export default function PagedAttentionVisualizer() {
  const [step, setStep] = useState(0)

  const blockSize = 4 // 每个block存4个token
  
  // 模拟3个请求
  const requests = [
    { id: 0, prompt: "What is AI?", tokens: ["What", "is", "AI", "?", "AI", "is", "..."] },
    { id: 1, prompt: "What is ML?", tokens: ["What", "is", "ML", "?", "ML", "is", "..."] },
    { id: 2, prompt: "Explain", tokens: ["Explain", "deep", "learning", ".", "..."] },
  ]

  // 计算每个步骤的block分配
  const getBlockAllocation = () => {
    const blocks: Block[] = []
    let blockIdCounter = 0

    if (step === 0) {
      // 初始状态：无分配
      return []
    }

    if (step >= 1) {
      // Request 0: "What is"
      blocks.push({
        id: blockIdCounter++,
        tokens: ["What", "is", "AI", "?"],
        owner: 0,
        shared: false
      })
    }

    if (step >= 2) {
      // Request 1: 共享 "What is" block
      blocks[0].shared = true
      blocks.push({
        id: blockIdCounter++,
        tokens: ["ML", "?", "", ""],
        owner: 1,
        shared: false
      })
    }

    if (step >= 3) {
      // Request 0 继续生成
      blocks.push({
        id: blockIdCounter++,
        tokens: ["AI", "is", "...", ""],
        owner: 0,
        shared: false
      })
    }

    if (step >= 4) {
      // Request 2
      blocks.push({
        id: blockIdCounter++,
        tokens: ["Explain", "deep", "learning", "."],
        owner: 2,
        shared: false
      })
    }

    if (step >= 5) {
      // Request 1 继续生成
      blocks.push({
        id: blockIdCounter++,
        tokens: ["ML", "is", "...", ""],
        owner: 1,
        shared: false
      })
    }

    return blocks
  }

  const blocks = getBlockAllocation()

  const getRequestBlocks = (requestId: number) => {
    return blocks.filter(b => b.owner === requestId || (b.shared && requestId <= 1))
  }

  const stepDescriptions = [
    "初始状态：内存池为空",
    "Request 0 开始：分配 Block 0 存储 'What is AI ?'",
    "Request 1 开始：共享 Block 0（相同前缀），分配 Block 1",
    "Request 0 继续生成：分配 Block 2",
    "Request 2 开始：分配 Block 3",
    "Request 1 继续生成：分配 Block 4",
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Database className="w-8 h-8 text-teal-600" />
        <h3 className="text-2xl font-bold text-slate-800">PagedAttention 内存管理</h3>
      </div>

      {/* 步骤控制 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="text-sm text-slate-600">当前步骤</div>
            <div className="text-xl font-bold text-teal-600">{step} / 5</div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
              className="px-4 py-2 bg-slate-600 text-white rounded disabled:opacity-50"
            >
              上一步
            </button>
            <button
              onClick={() => setStep(Math.min(5, step + 1))}
              disabled={step === 5}
              className="px-4 py-2 bg-teal-600 text-white rounded disabled:opacity-50"
            >
              下一步
            </button>
          </div>
        </div>
        <div className="text-sm text-slate-700 bg-teal-50 p-3 rounded border border-teal-200">
          {stepDescriptions[step]}
        </div>
      </div>

      {/* Block 内存池 */}
      <div className="mb-6 bg-white p-6 rounded-lg shadow">
        <div className="flex items-center gap-2 mb-4">
          <Box className="w-5 h-5 text-teal-600" />
          <h4 className="font-bold text-slate-800">Block 内存池 (Block Size = {blockSize} tokens)</h4>
        </div>

        <div className="grid grid-cols-6 gap-3">
          <AnimatePresence>
            {blocks.map((block) => (
              <motion.div
                key={block.id}
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                className={`p-3 rounded-lg border-2 ${
                  block.shared
                    ? 'border-purple-500 bg-purple-50'
                    : block.owner === 0
                    ? 'border-blue-500 bg-blue-50'
                    : block.owner === 1
                    ? 'border-green-500 bg-green-50'
                    : 'border-orange-500 bg-orange-50'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs font-bold text-slate-600">Block {block.id}</div>
                  {block.shared && (
                    <Share2 className="w-3 h-3 text-purple-600" />
                  )}
                </div>
                <div className="space-y-1">
                  {block.tokens.map((token, idx) => (
                    <div
                      key={idx}
                      className={`text-xs p-1 rounded text-center ${
                        token
                          ? 'bg-white font-mono'
                          : 'bg-slate-100 text-slate-400'
                      }`}
                    >
                      {token || '—'}
                    </div>
                  ))}
                </div>
                {block.shared && (
                  <div className="mt-2 text-[10px] text-purple-700 font-bold text-center">
                    共享
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* 空闲block */}
          {Array.from({ length: Math.max(0, 6 - blocks.length) }).map((_, idx) => (
            <div
              key={`empty-${idx}`}
              className="p-3 rounded-lg border-2 border-dashed border-slate-300 bg-slate-50"
            >
              <div className="text-xs text-slate-400 mb-2">空闲</div>
              <div className="space-y-1">
                {Array.from({ length: blockSize }).map((_, i) => (
                  <div key={i} className="h-5 bg-slate-200 rounded" />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 请求视图 */}
      <div className="grid grid-cols-3 gap-4">
        {requests.map((request) => {
          const requestBlocks = getRequestBlocks(request.id)
          const totalTokens = requestBlocks.reduce((sum, b) => sum + b.tokens.filter(t => t).length, 0)

          return (
            <div
              key={request.id}
              className={`p-4 rounded-lg border-2 ${
                request.id === 0
                  ? 'border-blue-300 bg-blue-50'
                  : request.id === 1
                  ? 'border-green-300 bg-green-50'
                  : 'border-orange-300 bg-orange-50'
              }`}
            >
              <div className="font-bold text-slate-800 mb-2">Request {request.id}</div>
              <div className="text-xs text-slate-600 mb-3 font-mono">"{request.prompt}"</div>

              {requestBlocks.length > 0 ? (
                <>
                  <div className="text-xs text-slate-600 mb-2">
                    使用的 Blocks: {requestBlocks.map(b => b.id).join(', ')}
                  </div>
                  <div className="flex gap-1 mb-2">
                    {requestBlocks.map((block) => (
                      <div
                        key={block.id}
                        className={`px-2 py-1 rounded text-xs font-bold ${
                          block.shared
                            ? 'bg-purple-200 text-purple-800'
                            : 'bg-white border border-slate-300'
                        }`}
                      >
                        {block.id}
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-slate-600">
                    总 tokens: {totalTokens}
                  </div>
                </>
              ) : (
                <div className="text-xs text-slate-400">尚未分配</div>
              )}
            </div>
          )
        })}
      </div>

      {/* 优势说明 */}
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-3">PagedAttention 优势</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-3 bg-green-50 rounded border border-green-200">
            <div className="font-bold text-green-800 mb-1">✓ 内存利用率高</div>
            <div className="text-sm text-slate-700">
              按需分配，用完立即回收，无需预留最大长度
            </div>
          </div>
          <div className="p-3 bg-blue-50 rounded border border-blue-200">
            <div className="font-bold text-blue-800 mb-1">✓ 支持前缀共享</div>
            <div className="text-sm text-slate-700">
              相同前缀的请求共享 Block，节省显存
            </div>
          </div>
          <div className="p-3 bg-purple-50 rounded border border-purple-200">
            <div className="font-bold text-purple-800 mb-1">✓ 动态分配</div>
            <div className="text-sm text-slate-700">
              非连续内存布局，避免碎片化
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-teal-50 border border-teal-200 rounded text-sm">
          <strong>性能提升</strong>: vLLM 使用 PagedAttention 后，吞吐量提升 2-3x，显存占用降低 50%
        </div>
      </div>
    </div>
  )
}

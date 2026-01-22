"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Layers } from 'lucide-react'

export default function AttentionMaskBuilder() {
  const [sequences, setSequences] = useState([
    { text: "Hello world", length: 3 },
    { text: "How are you today?", length: 6 },
  ])
  const maxLen = Math.max(...sequences.map(s => s.length))

  const buildMask = (actualLen: number, maxLen: number) => {
    return Array.from({ length: maxLen }, (_, i) => i < actualLen ? 1 : 0)
  }

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Layers className="w-5 h-5 text-blue-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Attention Mask 与 Padding 可视化
        </h3>
      </div>

      {/* 序列可视化 */}
      <div className="space-y-4">
        {sequences.map((seq, idx) => {
          const mask = buildMask(seq.length, maxLen)
          return (
            <div key={idx} className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700">
              <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">
                序列 {idx + 1}: "{seq.text}"
              </div>
              <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${maxLen}, 1fr)` }}>
                {mask.map((m, tidx) => (
                  <motion.div
                    key={tidx}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: tidx * 0.05 }}
                    className={`aspect-square rounded-lg flex items-center justify-center border-2 ${
                      m === 1
                        ? 'bg-green-100 dark:bg-green-900 border-green-400 dark:border-green-600'
                        : 'bg-red-100 dark:bg-red-900 border-red-400 dark:border-red-600'
                    }`}
                  >
                    {m === 1 ? (
                      <Eye className="w-4 h-4 text-green-700 dark:text-green-300" />
                    ) : (
                      <EyeOff className="w-4 h-4 text-red-700 dark:text-red-300" />
                    )}
                  </motion.div>
                ))}
              </div>
              <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                Attention Mask: [{mask.join(', ')}]
              </div>
            </div>
          )
        })}
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
        <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">工作原理</h4>
        <ul className="text-xs text-blue-800 dark:text-blue-200 space-y-1">
          <li>• <Eye className="w-3 h-3 inline" /> <strong>1</strong> = 真实 token，模型需要关注</li>
          <li>• <EyeOff className="w-3 h-3 inline" /> <strong>0</strong> = Padding token，模型忽略</li>
          <li>• Attention 计算时，mask=0 的位置分数会被设为 -∞，softmax 后变为 0</li>
        </ul>
      </div>
    </div>
  )
}

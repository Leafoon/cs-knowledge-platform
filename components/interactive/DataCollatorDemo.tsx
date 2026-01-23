'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { AlignLeft, Maximize2 } from 'lucide-react'

interface Sample {
  id: number
  text: string
  tokens: number[]
  length: number
}

export default function DataCollatorDemo() {
  const [paddingMode, setPaddingMode] = useState<'dynamic' | 'fixed'>('dynamic')
  
  const samples: Sample[] = [
    { id: 1, text: "Hello", tokens: [101, 7592, 102], length: 3 },
    { id: 2, text: "How are you?", tokens: [101, 2129, 2024, 2017, 1029, 102], length: 6 },
    { id: 3, text: "Good", tokens: [101, 2204, 102], length: 3 },
    { id: 4, text: "Machine learning", tokens: [101, 3698, 4083, 102], length: 4 }
  ]

  const maxLength = paddingMode === 'fixed' ? 512 : Math.max(...samples.map(s => s.length))
  
  const getPaddedTokens = (sample: Sample) => {
    const padded = [...sample.tokens]
    while (padded.length < maxLength) {
      padded.push(0) // PAD token
    }
    return padded
  }

  const getTotalTokens = () => {
    return paddingMode === 'fixed' 
      ? samples.length * 512
      : samples.reduce((sum, s) => sum + maxLength, 0)
  }

  const getWastedTokens = () => {
    const total = getTotalTokens()
    const actual = samples.reduce((sum, s) => sum + s.length, 0)
    return total - actual
  }

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-emerald-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <AlignLeft className="w-5 h-5 text-green-500" />
          DataCollator Padding 策略对比
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          动态 Padding vs 固定 Padding 的计算效率差异
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setPaddingMode('dynamic')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            paddingMode === 'dynamic'
              ? 'bg-green-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <AlignLeft className="w-4 h-4" />
            动态 Padding
          </div>
          <div className="text-xs mt-1 opacity-75">
            padding=&quot;longest&quot;
          </div>
        </button>
        <button
          onClick={() => setPaddingMode('fixed')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            paddingMode === 'fixed'
              ? 'bg-orange-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <Maximize2 className="w-4 h-4" />
            固定 Padding
          </div>
          <div className="text-xs mt-1 opacity-75">
            padding=&quot;max_length&quot;
          </div>
        </button>
      </div>

      {/* Samples Visualization */}
      <div className="space-y-3 mb-6">
        {samples.map((sample) => {
          const paddedTokens = getPaddedTokens(sample)
          return (
            <motion.div
              key={sample.id}
              layout
              className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  {sample.text}
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  长度: {sample.length}
                </span>
              </div>
              
              {/* Token Grid */}
              <div className="flex flex-wrap gap-1">
                {paddedTokens.slice(0, maxLength).map((token, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: idx * 0.01 }}
                    className={`px-2 py-1 rounded text-xs font-mono ${
                      token === 0
                        ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                        : token === 101 || token === 102
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                        : 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                    }`}
                  >
                    {token}
                  </motion.div>
                ))}
                {paddingMode === 'fixed' && paddedTokens.length > 20 && (
                  <div className="px-2 py-1 text-xs text-slate-400">... ({paddedTokens.length - 20} more)</div>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-blue-500">{maxLength}</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">Padding 长度</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-green-500">{getTotalTokens()}</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">总 Token 数</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-red-500">{getWastedTokens()}</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">浪费 Token</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-purple-500">
            {paddingMode === 'dynamic' ? '88%' : '0%'}
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400">计算节省</div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-blue-100 dark:bg-blue-900/30"></div>
          <span className="text-slate-600 dark:text-slate-400">特殊 Token ([CLS]/[SEP])</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-green-100 dark:bg-green-900/30"></div>
          <span className="text-slate-600 dark:text-slate-400">正常 Token</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-100 dark:bg-red-900/30"></div>
          <span className="text-slate-600 dark:text-slate-400">Padding Token</span>
        </div>
      </div>
    </div>
  )
}

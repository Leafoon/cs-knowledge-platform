"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, Info } from 'lucide-react'

export default function ConfigEditor() {
  const [config, setConfig] = useState({
    num_hidden_layers: 12,
    hidden_size: 768,
    num_attention_heads: 12,
    intermediate_size: 3072,
  })

  const updateConfig = (key: string, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  // 计算参数量（简化公式）
  const calculateParams = () => {
    const { num_hidden_layers, hidden_size, num_attention_heads, intermediate_size } = config
    const embeddingParams = 30522 * hidden_size // vocab_size * hidden_size
    const attentionParams = num_hidden_layers * (4 * hidden_size * hidden_size) // Q, K, V, O
    const ffnParams = num_hidden_layers * (2 * hidden_size * intermediate_size) // up + down
    const total = embeddingParams + attentionParams + ffnParams
    return (total / 1e6).toFixed(1) + 'M'
  }

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Settings className="w-5 h-5 text-purple-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          实时配置编辑器
        </h3>
      </div>

      {/* 配置参数滑块 */}
      <div className="space-y-6">
        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
              num_hidden_layers
            </span>
            <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
              {config.num_hidden_layers}
            </span>
          </label>
          <input
            type="range"
            min="1"
            max="24"
            value={config.num_hidden_layers}
            onChange={(e) => updateConfig('num_hidden_layers', parseInt(e.target.value))}
            className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
              hidden_size
            </span>
            <span className="text-lg font-bold text-green-600 dark:text-green-400">
              {config.hidden_size}
            </span>
          </label>
          <input
            type="range"
            min="256"
            max="1024"
            step="128"
            value={config.hidden_size}
            onChange={(e) => updateConfig('hidden_size', parseInt(e.target.value))}
            className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div>
          <label className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
              num_attention_heads
            </span>
            <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
              {config.num_attention_heads}
            </span>
          </label>
          <input
            type="range"
            min="4"
            max="16"
            value={config.num_attention_heads}
            onChange={(e) => updateConfig('num_attention_heads', parseInt(e.target.value))}
            className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
          {config.hidden_size % config.num_attention_heads !== 0 && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              ⚠️ hidden_size 必须能被 num_attention_heads 整除
            </p>
          )}
        </div>
      </div>

      {/* 计算结果 */}
      <motion.div
        key={JSON.stringify(config)}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="mt-6 p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 border border-blue-200 dark:border-blue-800"
      >
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">参数量估算</div>
            <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
              {calculateParams()}
            </div>
          </div>
          <div>
            <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">每个头的维度</div>
            <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
              {config.hidden_size / config.num_attention_heads}
            </div>
          </div>
        </div>
      </motion.div>

      {/* 生成的 config.json */}
      <div className="mt-6">
        <h4 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 mb-2">config.json</h4>
        <pre className="p-3 rounded-lg bg-neutral-900 dark:bg-neutral-800 border border-neutral-700 text-xs text-green-400 overflow-x-auto">
{`{
  "num_hidden_layers": ${config.num_hidden_layers},
  "hidden_size": ${config.hidden_size},
  "num_attention_heads": ${config.num_attention_heads},
  "intermediate_size": ${config.intermediate_size}
}`}
        </pre>
      </div>
    </div>
  )
}

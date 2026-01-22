"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Eye, ChevronDown, ChevronRight } from 'lucide-react'

interface OutputField {
  name: string
  shape: string
  dtype: string
  description: string
}

export default function ModelOutputInspector() {
  const [expandedFields, setExpandedFields] = useState<Set<string>>(new Set(['last_hidden_state']))

  const outputs: OutputField[] = [
    {
      name: 'last_hidden_state',
      shape: '[batch_size, sequence_length, hidden_size]',
      dtype: 'torch.FloatTensor',
      description: '最后一层的隐藏状态，包含序列中每个 token 的上下文表示'
    },
    {
      name: 'pooler_output',
      shape: '[batch_size, hidden_size]',
      dtype: 'torch.FloatTensor',
      description: '[CLS] token 的表示，经过线性层和 Tanh 激活函数处理'
    },
    {
      name: 'hidden_states',
      shape: 'Tuple of [batch_size, sequence_length, hidden_size]',
      dtype: 'tuple(torch.FloatTensor)',
      description: '所有层的隐藏状态（需设置 output_hidden_states=True），包含 embedding 层和每个 Transformer 层的输出'
    },
    {
      name: 'attentions',
      shape: 'Tuple of [batch_size, num_heads, sequence_length, sequence_length]',
      dtype: 'tuple(torch.FloatTensor)',
      description: '所有层的注意力权重（需设置 output_attentions=True），用于可视化模型关注的位置'
    },
  ]

  const toggleField = (name: string) => {
    setExpandedFields(prev => {
      const newSet = new Set(prev)
      if (newSet.has(name)) {
        newSet.delete(name)
      } else {
        newSet.add(name)
      }
      return newSet
    })
  }

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Eye className="w-5 h-5 text-green-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          模型输出结构探索器
        </h3>
      </div>

      <div className="space-y-3">
        {outputs.map((field, idx) => (
          <motion.div
            key={field.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="rounded-lg border border-neutral-200 dark:border-neutral-700 overflow-hidden"
          >
            <button
              onClick={() => toggleField(field.name)}
              className="w-full p-4 flex items-center justify-between bg-neutral-50 dark:bg-neutral-800 hover:bg-neutral-100 dark:hover:bg-neutral-750 transition-colors"
            >
              <div className="flex items-center gap-3">
                {expandedFields.has(field.name) ? (
                  <ChevronDown className="w-4 h-4 text-neutral-500" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-neutral-500" />
                )}
                <code className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                  {field.name}
                </code>
              </div>
              <span className="text-xs text-neutral-500 dark:text-neutral-400 font-mono">
                {field.dtype}
              </span>
            </button>

            <AnimatePresence>
              {expandedFields.has(field.name) && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 bg-white dark:bg-neutral-900 border-t border-neutral-200 dark:border-neutral-700">
                    <div className="mb-3">
                      <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">形状</div>
                      <code className="text-sm text-green-600 dark:text-green-400 font-mono">
                        {field.shape}
                      </code>
                    </div>
                    <div>
                      <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">描述</div>
                      <p className="text-sm text-neutral-700 dark:text-neutral-300">
                        {field.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* 使用示例 */}
      <div className="mt-6 p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700">
        <h4 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 mb-2">访问方式</h4>
        <pre className="text-xs text-neutral-700 dark:text-neutral-300 font-mono">
{`# 属性访问
outputs.last_hidden_state
outputs.pooler_output

# 字典访问
outputs["last_hidden_state"]

# 元组解包
last_hidden, pooler = outputs.to_tuple()[:2]`}
        </pre>
      </div>
    </div>
  )
}

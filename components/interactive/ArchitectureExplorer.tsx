"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, ArrowRight, Network } from 'lucide-react'

type ArchType = 'encoder' | 'decoder' | 'encoder-decoder'

export default function ArchitectureExplorer() {
  const [selected, setSelected] = useState<ArchType>('encoder')

  const architectures = {
    encoder: {
      name: 'Encoder-only (BERT)',
      color: 'blue',
      structure: ['Input', 'Embedding', 'Bi-directional Attention ×12', 'Pooling', 'Task Head'],
      attention: '双向',
      pretrain: 'MLM (掩码语言模型)',
      tasks: ['文本分类', 'NER', '问答'],
      models: ['BERT', 'RoBERTa', 'DeBERTa']
    },
    decoder: {
      name: 'Decoder-only (GPT)',
      color: 'purple',
      structure: ['Input', 'Embedding', 'Causal Attention ×12', 'LM Head'],
      attention: '单向（因果掩码）',
      pretrain: 'CLM (自回归语言模型)',
      tasks: ['文本生成', '对话', '代码生成'],
      models: ['GPT-2', 'LLaMA', 'Mistral']
    },
    'encoder-decoder': {
      name: 'Encoder-Decoder (T5)',
      color: 'green',
      structure: ['Encoder (Bi-dir ×6)', 'Decoder (Causal + Cross-Attn ×6)', 'LM Head'],
      attention: '编码器双向 + 解码器单向 + 交叉注意力',
      pretrain: 'Span Corruption',
      tasks: ['翻译', '摘要', 'Seq2Seq'],
      models: ['T5', 'BART', 'mT5']
    }
  }

  const arch = architectures[selected]
  const colorClasses = {
    blue: 'bg-blue-500 text-blue-50',
    purple: 'bg-purple-500 text-purple-50',
    green: 'bg-green-500 text-green-50'
  }

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Network className="w-5 h-5 text-indigo-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Transformer 架构对比
        </h3>
      </div>

      {/* 架构选择 */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {(Object.keys(architectures) as ArchType[]).map((key) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selected === key
                ? `border-${architectures[key].color}-500 bg-${architectures[key].color}-50 dark:bg-${architectures[key].color}-950`
                : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300'
            }`}
          >
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {architectures[key].name}
            </div>
          </button>
        ))}
      </div>

      {/* 架构可视化 */}
      <motion.div
        key={selected}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex flex-col gap-3">
          {arch.structure.map((layer, idx) => (
            <React.Fragment key={idx}>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={`p-4 rounded-lg ${colorClasses[arch.color as keyof typeof colorClasses]} flex items-center justify-center font-semibold`}
              >
                {layer}
              </motion.div>
              {idx < arch.structure.length - 1 && (
                <ArrowRight className="w-5 h-5 text-neutral-400 self-center rotate-90" />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* 详细信息 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-neutral-50 dark:bg-neutral-800">
            <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">注意力机制</div>
            <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">{arch.attention}</div>
          </div>
          <div className="p-3 rounded-lg bg-neutral-50 dark:bg-neutral-800">
            <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">预训练任务</div>
            <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">{arch.pretrain}</div>
          </div>
        </div>

        <div className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800">
          <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">代表模型</div>
          <div className="flex flex-wrap gap-2">
            {arch.models.map((m, i) => (
              <span key={i} className="px-2 py-1 rounded bg-neutral-200 dark:bg-neutral-700 text-xs font-medium">
                {m}
              </span>
            ))}
          </div>
        </div>

        <div className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800">
          <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">适用任务</div>
          <div className="flex flex-wrap gap-2">
            {arch.tasks.map((t, i) => (
              <span key={i} className="px-2 py-1 rounded bg-neutral-200 dark:bg-neutral-700 text-xs">
                {t}
              </span>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  )
}

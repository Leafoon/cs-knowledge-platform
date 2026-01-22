"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, Type, Hash, Sparkles } from 'lucide-react'

export default function TokenizationVisualizer() {
  const [inputText, setInputText] = useState("Hugging Face is amazing!")
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<'wordpiece' | 'bpe' | 'sentencepiece'>('wordpiece')
  
  // 模拟 tokenization 结果
  const tokenize = (text: string, algorithm: string) => {
    const normalized = text.toLowerCase()
    
    if (algorithm === 'wordpiece') {
      // BERT WordPiece 风格
      return [
        { token: '[CLS]', id: 101, isSpecial: true, color: '#8b5cf6' },
        { token: 'hugging', id: 17662, isSpecial: false, color: '#3b82f6' },
        { token: 'face', id: 2227, isSpecial: false, color: '#3b82f6' },
        { token: 'is', id: 2003, isSpecial: false, color: '#10b981' },
        { token: 'amazing', id: 6429, isSpecial: false, color: '#f59e0b' },
        { token: '!', id: 999, isSpecial: false, color: '#ef4444' },
        { token: '[SEP]', id: 102, isSpecial: true, color: '#8b5cf6' },
      ]
    } else if (algorithm === 'bpe') {
      // GPT-2 BPE 风格
      return [
        { token: 'H', id: 39, isSpecial: false, color: '#3b82f6' },
        { token: 'ugg', id: 2667, isSpecial: false, color: '#3b82f6' },
        { token: 'ing', id: 278, isSpecial: false, color: '#3b82f6' },
        { token: 'ĠFace', id: 15399, isSpecial: false, color: '#10b981' },
        { token: 'Ġis', id: 318, isSpecial: false, color: '#10b981' },
        { token: 'Ġamazing', id: 4998, isSpecial: false, color: '#f59e0b' },
        { token: '!', id: 0, isSpecial: false, color: '#ef4444' },
      ]
    } else {
      // SentencePiece 风格
      return [
        { token: '▁Hugging', id: 23499, isSpecial: false, color: '#3b82f6' },
        { token: '▁Face', id: 10552, isSpecial: false, color: '#10b981' },
        { token: '▁is', id: 19, isSpecial: false, color: '#10b981' },
        { token: '▁amazing', id: 6821, isSpecial: false, color: '#f59e0b' },
        { token: '!', id: 55, isSpecial: false, color: '#ef4444' },
      ]
    }
  }

  const tokens = tokenize(inputText, selectedAlgorithm)

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="w-5 h-5 text-yellow-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          实时 Tokenization 可视化
        </h3>
      </div>

      {/* 算法选择 */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
          Tokenization 算法
        </label>
        <div className="grid grid-cols-3 gap-2">
          {[
            { value: 'wordpiece' as const, label: 'WordPiece', desc: 'BERT' },
            { value: 'bpe' as const, label: 'BPE', desc: 'GPT-2' },
            { value: 'sentencepiece' as const, label: 'SentencePiece', desc: 'T5' },
          ].map((algo) => (
            <button
              key={algo.value}
              onClick={() => setSelectedAlgorithm(algo.value)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedAlgorithm === algo.value
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                  : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300'
              }`}
            >
              <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                {algo.label}
              </div>
              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                {algo.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 输入文本 */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
          输入文本
        </label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full px-4 py-2 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="输入任意文本..."
        />
      </div>

      {/* Tokenization 流程 */}
      <div className="space-y-4">
        {/* 步骤1: 原始文本 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700"
        >
          <div className="flex items-center gap-2 mb-2">
            <Type className="w-4 h-4 text-neutral-600 dark:text-neutral-400" />
            <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
              步骤 1: 原始文本
            </span>
          </div>
          <div className="text-base font-mono text-neutral-900 dark:text-neutral-100">
            {inputText}
          </div>
        </motion.div>

        <div className="flex justify-center">
          <ChevronRight className="w-6 h-6 text-neutral-400" />
        </div>

        {/* 步骤2: Tokens */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700"
        >
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
              步骤 2: Subword Tokens
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            <AnimatePresence mode="popLayout">
              {tokens.map((token, idx) => (
                <motion.div
                  key={`${token.token}-${idx}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`px-3 py-1.5 rounded-md font-mono text-sm border-2 ${
                    token.isSpecial
                      ? 'bg-purple-100 dark:bg-purple-900 border-purple-300 dark:border-purple-700 text-purple-900 dark:text-purple-100'
                      : 'bg-white dark:bg-neutral-700 border-neutral-200 dark:border-neutral-600'
                  }`}
                  style={{
                    color: token.isSpecial ? undefined : token.color,
                  }}
                >
                  {token.token}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </motion.div>

        <div className="flex justify-center">
          <ChevronRight className="w-6 h-6 text-neutral-400" />
        </div>

        {/* 步骤3: Token IDs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700"
        >
          <div className="flex items-center gap-2 mb-3">
            <Hash className="w-4 h-4 text-green-500" />
            <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
              步骤 3: Token IDs
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            <AnimatePresence mode="popLayout">
              {tokens.map((token, idx) => (
                <motion.div
                  key={`id-${idx}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ delay: idx * 0.05 + 0.1 }}
                  className="px-3 py-1.5 rounded-md font-mono text-sm bg-green-100 dark:bg-green-900 border-2 border-green-300 dark:border-green-700 text-green-900 dark:text-green-100"
                >
                  {token.id}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
          <div className="mt-3 text-xs text-neutral-500 dark:text-neutral-400">
            张量形状: [1, {tokens.length}]
          </div>
        </motion.div>
      </div>

      {/* 统计信息 */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
          <div className="text-xs text-blue-600 dark:text-blue-400 mb-1">Token 数量</div>
          <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">{tokens.length}</div>
        </div>
        <div className="p-3 rounded-lg bg-purple-50 dark:bg-purple-950 border border-purple-200 dark:border-purple-800">
          <div className="text-xs text-purple-600 dark:text-purple-400 mb-1">特殊 Token</div>
          <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
            {tokens.filter(t => t.isSpecial).length}
          </div>
        </div>
        <div className="p-3 rounded-lg bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800">
          <div className="text-xs text-green-600 dark:text-green-400 mb-1">压缩率</div>
          <div className="text-2xl font-bold text-green-900 dark:text-green-100">
            {(inputText.split(' ').length / tokens.filter(t => !t.isSpecial).length).toFixed(2)}x
          </div>
        </div>
      </div>

      {/* 算法说明 */}
      <div className="mt-6 p-4 rounded-lg bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800">
        <h4 className="text-sm font-semibold text-amber-900 dark:text-amber-100 mb-2">
          {selectedAlgorithm === 'wordpiece' && 'WordPiece 特点'}
          {selectedAlgorithm === 'bpe' && 'BPE 特点'}
          {selectedAlgorithm === 'sentencepiece' && 'SentencePiece 特点'}
        </h4>
        <ul className="text-xs text-amber-800 dark:text-amber-200 space-y-1">
          {selectedAlgorithm === 'wordpiece' && (
            <>
              <li>• 使用 [CLS] 和 [SEP] 特殊标记</li>
              <li>• 子词使用 ## 前缀（如 ##ing）</li>
              <li>• 贪婪最长匹配策略</li>
            </>
          )}
          {selectedAlgorithm === 'bpe' && (
            <>
              <li>• 使用 Ġ 表示空格（Unicode U+0120）</li>
              <li>• Byte-level 编码，支持任意 Unicode</li>
              <li>• 从字符对频率统计构建词汇表</li>
            </>
          )}
          {selectedAlgorithm === 'sentencepiece' && (
            <>
              <li>• 使用 ▁ 表示词开头（空格编码）</li>
              <li>• 语言无关，无需预分词</li>
              <li>• 可逆性：decode(encode(text)) == text</li>
            </>
          )}
        </ul>
      </div>
    </div>
  )
}

"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Code, Zap, Globe, TrendingUp } from 'lucide-react'

interface AlgorithmData {
  name: string
  icon: React.ReactNode
  color: string
  bgColor: string
  borderColor: string
  description: string
  mechanism: string
  examples: { input: string; output: string[] }[]
  pros: string[]
  cons: string[]
  usedBy: string[]
  vocabSize: string
}

export default function TokenAlgorithmComparison() {
  const [selectedAlgo, setSelectedAlgo] = useState<'wordpiece' | 'bpe' | 'unigram'>('wordpiece')

  const algorithms: Record<'wordpiece' | 'bpe' | 'unigram', AlgorithmData> = {
    wordpiece: {
      name: 'WordPiece',
      icon: <Code className="w-5 h-5" />,
      color: 'text-blue-600 dark:text-blue-400',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      borderColor: 'border-blue-300 dark:border-blue-700',
      description: 'Google 提出的贪婪最长匹配算法，用于 BERT 系列模型',
      mechanism: '从左到右贪婪匹配最长的词汇表子串，如果无法匹配则拆分为更小的子词（添加 ## 前缀）',
      examples: [
        {
          input: 'playing',
          output: ['playing']
        },
        {
          input: 'unhappiness',
          output: ['un', '##hap', '##piness']
        },
        {
          input: 'Huggingface',
          output: ['hugging', '##face']
        }
      ],
      pros: [
        '简单高效，实现容易',
        '通过 ## 前缀明确标记子词边界',
        'OOV 鲁棒性强（可降级到字符级）',
        '预训练模型丰富'
      ],
      cons: [
        '依赖预分词（空格分隔）',
        '英文偏向，多语言支持有限',
        '词汇表构建需要语言模型似然'
      ],
      usedBy: ['BERT', 'DistilBERT', 'ELECTRA', 'MobileBERT'],
      vocabSize: '~30K'
    },
    bpe: {
      name: 'Byte-Pair Encoding (BPE)',
      icon: <Zap className="w-5 h-5" />,
      color: 'text-purple-600 dark:text-purple-400',
      bgColor: 'bg-purple-50 dark:bg-purple-950',
      borderColor: 'border-purple-300 dark:border-purple-700',
      description: '基于统计的自底向上合并算法，原用于数据压缩',
      mechanism: '从字符级开始，迭代合并最频繁的字符对，直到达到目标词汇表大小',
      examples: [
        {
          input: 'Hello, world!',
          output: ['Hello', ',', 'Ġworld', '!']
        },
        {
          input: 'lowercase',
          output: ['lower', 'case']
        },
        {
          input: 'café',
          output: ['c', 'af', 'Ã', '©']
        }
      ],
      pros: [
        'Byte-level 变体支持任意 Unicode',
        '无 [UNK] token（所有字符可编码）',
        '训练过程直观，基于频率统计',
        '生成质量优异'
      ],
      cons: [
        '词汇表可能包含无意义的字符组合',
        '空格处理需特殊字符（Ġ）',
        '不同实现细节差异大'
      ],
      usedBy: ['GPT-2', 'GPT-3', 'RoBERTa', 'BART', 'CodeGen'],
      vocabSize: '~50K'
    },
    unigram: {
      name: 'Unigram Language Model',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-green-600 dark:text-green-400',
      bgColor: 'bg-green-50 dark:bg-green-950',
      borderColor: 'border-green-300 dark:border-green-700',
      description: '基于概率模型的分词，使用 EM 算法优化',
      mechanism: '将分词视为概率问题，每个 token 有出现概率，用 Viterbi 算法找最优路径',
      examples: [
        {
          input: 'unhappiness',
          output: ['▁un', 'happiness']
        },
        {
          input: 'hello world',
          output: ['▁hello', '▁world']
        },
        {
          input: 'BERT vs GPT',
          output: ['▁BERT', '▁vs', '▁G', 'PT']
        }
      ],
      pros: [
        '多种分词方案，选择最优',
        '词汇表更紧凑高效',
        '理论基础扎实（概率建模）',
        '适合资源受限场景'
      ],
      cons: [
        '训练复杂度高（EM 算法）',
        '推理速度略慢（动态规划）',
        '可解释性不如 BPE'
      ],
      usedBy: ['XLNet', 'ALBERT', 'T5', 'mBART'],
      vocabSize: '~32K'
    }
  }

  const currentAlgo = algorithms[selectedAlgo]

  return (
    <div className="my-8 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-6">
        <Globe className="w-5 h-5 text-indigo-500" />
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Tokenization 算法对比
        </h3>
      </div>

      {/* 算法选择卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {Object.entries(algorithms).map(([key, algo]) => (
          <motion.button
            key={key}
            onClick={() => setSelectedAlgo(key as any)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`p-4 rounded-lg border-2 text-left transition-all ${
              selectedAlgo === key
                ? `${algo.borderColor} ${algo.bgColor}`
                : 'border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800 hover:border-neutral-300'
            }`}
          >
            <div className={`flex items-center gap-2 mb-2 ${selectedAlgo === key ? algo.color : 'text-neutral-600 dark:text-neutral-400'}`}>
              {algo.icon}
              <span className="font-semibold">{algo.name}</span>
            </div>
            <p className="text-xs text-neutral-600 dark:text-neutral-400 line-clamp-2">
              {algo.description}
            </p>
          </motion.button>
        ))}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedAlgo}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* 机制说明 */}
        <div className={`p-4 rounded-lg ${currentAlgo.bgColor} border ${currentAlgo.borderColor}`}>
          <h4 className={`text-sm font-semibold ${currentAlgo.color} mb-2`}>核心机制</h4>
          <p className="text-sm text-neutral-700 dark:text-neutral-300">
            {currentAlgo.mechanism}
          </p>
        </div>

        {/* 示例 */}
        <div>
          <h4 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 mb-3">
            Tokenization 示例
          </h4>
          <div className="space-y-3">
            {currentAlgo.examples.map((example, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700"
              >
                <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">输入</div>
                <div className="font-mono text-sm text-neutral-900 dark:text-neutral-100 mb-3">
                  &quot;{example.input}&quot;
                </div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400 mb-2">输出</div>
                <div className="flex flex-wrap gap-2">
                  {example.output.map((token, tidx) => (
                    <span
                      key={tidx}
                      className={`px-2 py-1 rounded font-mono text-xs ${currentAlgo.bgColor} ${currentAlgo.borderColor} border`}
                    >
                      {token}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 优缺点对比 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 优点 */}
          <div className="p-4 rounded-lg bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-800">
            <h4 className="text-sm font-semibold text-emerald-900 dark:text-emerald-100 mb-3">
              ✅ 优点
            </h4>
            <ul className="space-y-2">
              {currentAlgo.pros.map((pro, idx) => (
                <li key={idx} className="text-xs text-emerald-800 dark:text-emerald-200 flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5">•</span>
                  <span>{pro}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* 缺点 */}
          <div className="p-4 rounded-lg bg-rose-50 dark:bg-rose-950 border border-rose-200 dark:border-rose-800">
            <h4 className="text-sm font-semibold text-rose-900 dark:text-rose-100 mb-3">
              ⚠️ 缺点
            </h4>
            <ul className="space-y-2">
              {currentAlgo.cons.map((con, idx) => (
                <li key={idx} className="text-xs text-rose-800 dark:text-rose-200 flex items-start gap-2">
                  <span className="text-rose-500 mt-0.5">•</span>
                  <span>{con}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* 应用模型 */}
        <div className="p-4 rounded-lg bg-neutral-50 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              使用该算法的模型
            </h4>
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              词汇表大小: {currentAlgo.vocabSize}
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {currentAlgo.usedBy.map((model, idx) => (
              <span
                key={idx}
                className="px-3 py-1 rounded-full bg-neutral-200 dark:bg-neutral-700 text-xs font-medium text-neutral-700 dark:text-neutral-300"
              >
                {model}
              </span>
            ))}
          </div>
        </div>
      </motion.div>

      {/* 快速对比表 */}
      <div className="mt-8 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-neutral-200 dark:border-neutral-700">
              <th className="text-left py-3 px-4 font-semibold text-neutral-900 dark:text-neutral-100">特性</th>
              <th className="text-center py-3 px-4 font-semibold text-blue-600 dark:text-blue-400">WordPiece</th>
              <th className="text-center py-3 px-4 font-semibold text-purple-600 dark:text-purple-400">BPE</th>
              <th className="text-center py-3 px-4 font-semibold text-green-600 dark:text-green-400">Unigram</th>
            </tr>
          </thead>
          <tbody className="text-xs">
            <tr className="border-b border-neutral-100 dark:border-neutral-800">
              <td className="py-2 px-4 text-neutral-600 dark:text-neutral-400">合并策略</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">贪婪最长匹配</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">频率最高字符对</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">最大似然概率</td>
            </tr>
            <tr className="border-b border-neutral-100 dark:border-neutral-800">
              <td className="py-2 px-4 text-neutral-600 dark:text-neutral-400">子词标记</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">## 前缀</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">Ġ 空格</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">▁ 词开头</td>
            </tr>
            <tr className="border-b border-neutral-100 dark:border-neutral-800">
              <td className="py-2 px-4 text-neutral-600 dark:text-neutral-400">OOV 处理</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">[UNK] token</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">Byte-level</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">概率分解</td>
            </tr>
            <tr>
              <td className="py-2 px-4 text-neutral-600 dark:text-neutral-400">训练复杂度</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">中</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">低</td>
              <td className="py-2 px-4 text-center text-neutral-700 dark:text-neutral-300">高</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

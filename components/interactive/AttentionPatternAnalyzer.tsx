'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type PatternType = 'uniform' | 'diagonal' | 'vertical' | 'block' | 'custom'

interface AnalysisResult {
  entropy: number
  sparsity: number
  maxAttention: number
  avgAttention: number
  interpretation: string
}

export default function AttentionPatternAnalyzer() {
  const [gridSize] = useState(12)
  const [pattern, setPattern] = useState<PatternType>('diagonal')
  const [attentionMatrix, setAttentionMatrix] = useState<number[][]>(() =>
    generatePattern('diagonal', 12)
  )
  const [hoveredCell, setHoveredCell] = useState<{i: number; j: number} | null>(null)

  // 生成不同的注意力模式
  function generatePattern(type: PatternType, size: number): number[][] {
    const matrix: number[][] = Array(size).fill(0).map(() => Array(size).fill(0))

    switch (type) {
      case 'uniform':
        // 均匀注意力
        for (let i = 0; i < size; i++) {
          for (let j = 0; j <= i; j++) {
            matrix[i][j] = 1.0 / (i + 1)
          }
        }
        break

      case 'diagonal':
        // 局部注意力（对角线）
        for (let i = 0; i < size; i++) {
          for (let j = 0; j <= i; j++) {
            const distance = i - j
            matrix[i][j] = Math.exp(-distance / 2)
          }
        }
        break

      case 'vertical':
        // 垂直条纹（关注特定 token）
        const keyPositions = [0, 3, 7]
        for (let i = 0; i < size; i++) {
          for (let j = 0; j <= i; j++) {
            matrix[i][j] = keyPositions.includes(j) ? 0.9 : 0.1
          }
          // 归一化
          const sum = matrix[i].reduce((a, b) => a + b, 0)
          for (let j = 0; j <= i; j++) {
            matrix[i][j] /= sum
          }
        }
        break

      case 'block':
        // 块状注意力
        for (let i = 0; i < size; i++) {
          const block = Math.floor(i / 3)
          for (let j = 0; j <= i; j++) {
            const jBlock = Math.floor(j / 3)
            matrix[i][j] = block === jBlock ? 0.8 : 0.2
          }
          // 归一化
          const sum = matrix[i].reduce((a, b) => a + b, 0)
          for (let j = 0; j <= i; j++) {
            matrix[i][j] /= sum
          }
        }
        break

      case 'custom':
        // 随机模式
        for (let i = 0; i < size; i++) {
          for (let j = 0; j <= i; j++) {
            matrix[i][j] = Math.random()
          }
          // 归一化
          const sum = matrix[i].reduce((a, b) => a + b, 0)
          for (let j = 0; j <= i; j++) {
            matrix[i][j] /= sum
          }
        }
        break
    }

    return matrix
  }

  // 分析注意力模式
  function analyzePattern(matrix: number[][]): AnalysisResult {
    let totalEntropy = 0
    let nonZeroCount = 0
    let maxAttn = 0
    let sumAttn = 0
    let validCells = 0

    for (let i = 0; i < matrix.length; i++) {
      // 计算每行的熵
      let rowEntropy = 0
      for (let j = 0; j <= i; j++) {
        const p = matrix[i][j]
        if (p > 0) {
          rowEntropy -= p * Math.log2(p)
          nonZeroCount++
          sumAttn += p
          validCells++
        }
        maxAttn = Math.max(maxAttn, p)
      }
      totalEntropy += rowEntropy
    }

    const avgEntropy = totalEntropy / matrix.length
    const sparsity = 1 - nonZeroCount / validCells
    const avgAttention = validCells > 0 ? sumAttn / validCells : 0

    // 解释
    let interpretation = ''
    if (avgEntropy < 1.5) {
      interpretation = '集中注意力：模型强烈关注少数 tokens'
    } else if (avgEntropy < 3.0) {
      interpretation = '局部注意力：模型关注附近的 tokens'
    } else {
      interpretation = '分散注意力：模型均匀关注所有 tokens'
    }

    return {
      entropy: avgEntropy,
      sparsity,
      maxAttention: maxAttn,
      avgAttention,
      interpretation
    }
  }

  const analysis = analyzePattern(attentionMatrix)

  // 改变模式
  const changePattern = (newPattern: PatternType) => {
    setPattern(newPattern)
    setAttentionMatrix(generatePattern(newPattern, gridSize))
  }

  // 获取单元格颜色
  const getCellColor = (value: number) => {
    if (value === 0) return 'rgb(240, 240, 240)'
    // 从白色到深紫色
    const intensity = value
    return `rgba(168, 85, 247, ${0.1 + intensity * 0.9})`
  }

  const tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'slept', 'all', 'day', 'long', '.']

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">注意力模式分析工具</h3>
        <p className="text-gray-600">可视化和分析不同类型的注意力模式，理解模型如何处理序列</p>
      </div>

      {/* 模式选择器 */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <h4 className="text-lg font-bold mb-4">选择注意力模式</h4>
        <div className="grid grid-cols-5 gap-3">
          {[
            { type: 'uniform' as PatternType, name: '均匀注意力', desc: '平等关注所有 tokens' },
            { type: 'diagonal' as PatternType, name: '局部注意力', desc: '关注附近 tokens' },
            { type: 'vertical' as PatternType, name: '关键 Token', desc: '关注特定位置' },
            { type: 'block' as PatternType, name: '块状注意力', desc: '分块处理' },
            { type: 'custom' as PatternType, name: '随机模式', desc: '随机生成' }
          ].map(({ type, name, desc }) => (
            <motion.button
              key={type}
              onClick={() => changePattern(type)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                pattern === type
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="font-semibold text-sm">{name}</div>
              <div className="text-xs text-gray-500 mt-1">{desc}</div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* 主可视化区域 */}
      <div className="grid grid-cols-3 gap-6">
        {/* 左侧：热力图 */}
        <div className="col-span-2 bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h4 className="text-lg font-bold mb-4">注意力热力图</h4>
          
          <div className="relative">
            {/* Y轴标签 (Query Tokens) */}
            <div className="flex">
              <div className="w-16 flex flex-col justify-around text-right pr-2 text-xs">
                {tokens.map((token, i) => (
                  <div key={i} className="h-8 flex items-center justify-end font-mono">
                    {token}
                  </div>
                ))}
              </div>

              {/* 热力图网格 */}
              <div className="flex-1">
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
                  {attentionMatrix.map((row, i) =>
                    row.map((value, j) => (
                      <motion.div
                        key={`${i}-${j}`}
                        className="aspect-square border border-gray-200 cursor-pointer relative group"
                        style={{
                          backgroundColor: getCellColor(value),
                          opacity: j > i ? 0.3 : 1 // 未来 tokens 半透明
                        }}
                        onMouseEnter={() => setHoveredCell({ i, j })}
                        onMouseLeave={() => setHoveredCell(null)}
                        whileHover={{ scale: 1.1, zIndex: 10 }}
                      >
                        {hoveredCell?.i === i && hoveredCell?.j === j && (
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap z-20">
                            {tokens[i]} → {tokens[j]}: {value.toFixed(3)}
                          </div>
                        )}
                      </motion.div>
                    ))
                  )}
                </div>

                {/* X轴标签 (Key Tokens) */}
                <div className="flex justify-around mt-2 text-xs font-mono">
                  {tokens.map((token, i) => (
                    <div key={i} className="transform -rotate-45 origin-top-left">
                      {token}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 颜色图例 */}
            <div className="mt-4 flex items-center gap-4">
              <div className="text-sm font-semibold">注意力强度:</div>
              <div className="flex-1 h-6 rounded-lg" style={{
                background: 'linear-gradient(to right, rgb(240, 240, 240), rgba(168, 85, 247, 1))'
              }} />
              <div className="flex gap-4 text-xs text-gray-600">
                <span>0.0</span>
                <span>0.5</span>
                <span>1.0</span>
              </div>
            </div>
          </div>
        </div>

        {/* 右侧：分析结果 */}
        <div className="space-y-4">
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
            <h4 className="text-lg font-bold mb-4">模式分析</h4>
            
            <div className="space-y-4">
              {/* 熵 */}
              <div>
                <div className="text-sm text-gray-600 mb-1">信息熵</div>
                <div className="text-3xl font-bold text-purple-600">
                  {analysis.entropy.toFixed(2)}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all"
                    style={{ width: `${Math.min(analysis.entropy / 4 * 100, 100)}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  0 = 完全集中, 4+ = 完全均匀
                </div>
              </div>

              {/* 稀疏性 */}
              <div>
                <div className="text-sm text-gray-600 mb-1">稀疏度</div>
                <div className="text-3xl font-bold text-blue-600">
                  {(analysis.sparsity * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${analysis.sparsity * 100}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  高稀疏度 = 更高效的计算
                </div>
              </div>

              {/* 最大注意力 */}
              <div>
                <div className="text-sm text-gray-600 mb-1">最大注意力权重</div>
                <div className="text-3xl font-bold text-pink-600">
                  {(analysis.maxAttention * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-pink-500 h-2 rounded-full transition-all"
                    style={{ width: `${analysis.maxAttention * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* 解释 */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
            <h5 className="text-sm font-bold text-purple-900 mb-2">💡 模式解释</h5>
            <p className="text-sm text-purple-800">{analysis.interpretation}</p>
          </div>

          {/* 应用场景 */}
          <div className="bg-white rounded-lg shadow border p-4">
            <h5 className="text-sm font-bold mb-3">典型应用场景</h5>
            <div className="space-y-2 text-xs">
              {pattern === 'uniform' && (
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-semibold">填空任务 (MLM)</div>
                  <div className="text-gray-600">BERT 等双向模型</div>
                </div>
              )}
              {pattern === 'diagonal' && (
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-semibold">文本生成</div>
                  <div className="text-gray-600">GPT 等自回归模型</div>
                </div>
              )}
              {pattern === 'vertical' && (
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-semibold">指代消解</div>
                  <div className="text-gray-600">关注代词指向的实体</div>
                </div>
              )}
              {pattern === 'block' && (
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-semibold">文档分块处理</div>
                  <div className="text-gray-600">Longformer 局部窗口</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 数学公式 */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <h4 className="text-lg font-bold mb-4">注意力机制数学公式</h4>
        
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-sm font-semibold mb-2">标准注意力</div>
            <div className="font-mono text-sm bg-white p-3 rounded border overflow-x-auto">
              Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-sm font-semibold mb-2">信息熵（衡量注意力集中度）</div>
            <div className="font-mono text-sm bg-white p-3 rounded border overflow-x-auto">
              H = -∑ p<sub>i</sub> log<sub>2</sub>(p<sub>i</sub>)
            </div>
            <div className="text-xs text-gray-600 mt-2">
              其中 p<sub>i</sub> 是第 i 个 token 的注意力权重
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-sm font-semibold mb-2">稀疏度</div>
            <div className="font-mono text-sm bg-white p-3 rounded border overflow-x-auto">
              Sparsity = 1 - (non-zero elements) / (total elements)
            </div>
          </div>
        </div>
      </div>

      {/* 交互提示 */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-bold text-blue-900 mb-2">💡 使用提示</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• 悬停在热力图单元格上查看具体数值</li>
          <li>• 切换不同模式观察注意力分布变化</li>
          <li>• 观察对角线 = 自注意力强度</li>
          <li>• 垂直条纹 = 某个 token 被广泛关注（如 [CLS]）</li>
        </ul>
      </div>

      {/* 代码示例 */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <div className="text-xs text-gray-300 mb-2">使用 BertViz 可视化真实注意力</div>
        <pre className="text-sm text-gray-100">
          <code>{`from bertviz import head_view, model_view
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 可视化所有注意力头
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attention = outputs.attentions

head_view(attention, tokens)  # 单层多头
model_view(attention, tokens)  # 所有层`}</code>
        </pre>
      </div>
    </div>
  )
}

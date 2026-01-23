'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type Strategy = 'standard' | 'alibi' | 'rope' | 'sparse' | 'rag'

interface StrategyInfo {
  name: string
  description: string
  complexity: string
  maxLength: string
  advantages: string[]
  disadvantages: string[]
  color: string
}

const strategies: Record<Strategy, StrategyInfo> = {
  standard: {
    name: 'Standard Attention',
    description: '标准的全注意力机制，每个 token 关注所有其他 token',
    complexity: 'O(n²)',
    maxLength: '~2048 tokens',
    advantages: ['完整的上下文信息', '简单直接', '性能最优（短序列）'],
    disadvantages: ['二次复杂度', '内存消耗巨大', '难以扩展到长序列'],
    color: 'blue'
  },
  alibi: {
    name: 'ALiBi (Linear Biases)',
    description: '通过在注意力分数上添加线性偏置，无需位置编码即可外推到更长序列',
    complexity: 'O(n²)',
    maxLength: '∞ (理论上)',
    advantages: ['零外推能力', '无位置编码参数', '训练短推理长'],
    disadvantages: ['仍是二次复杂度', '需要调整偏置斜率'],
    color: 'green'
  },
  rope: {
    name: 'RoPE + Interpolation',
    description: '旋转位置编码 + 位置插值，通过缩放位置索引扩展上下文窗口',
    complexity: 'O(n²)',
    maxLength: '8K-128K tokens',
    advantages: ['平滑扩展', 'NTK-aware 优化', '兼容性好'],
    disadvantages: ['需要微调插值因子', '极长序列性能下降'],
    color: 'purple'
  },
  sparse: {
    name: 'Sparse Attention',
    description: '稀疏注意力模式（局部+全局+随机），大幅降低计算复杂度',
    complexity: 'O(n log n) or O(n)',
    maxLength: '16K-64K tokens',
    advantages: ['线性/对数复杂度', '内存高效', '适合超长文档'],
    disadvantages: ['丢失部分上下文', '实现复杂', '需要特殊设计'],
    color: 'orange'
  },
  rag: {
    name: 'Retrieval-Augmented',
    description: '检索外部知识库，动态扩展上下文，无需将所有信息编码到参数',
    complexity: 'O(n² + k·m)',
    maxLength: 'Unlimited (检索)',
    advantages: ['无限上下文', '知识可更新', '降低参数量'],
    disadvantages: ['依赖检索质量', '增加推理延迟', '工程复杂度高'],
    color: 'pink'
  }
}

export default function LongContextStrategies() {
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy>('standard')
  const [sequenceLength, setSequenceLength] = useState(1024)

  const strategy = strategies[selectedStrategy]

  // 计算内存占用（简化模型）
  const calculateMemory = (length: number, strat: Strategy) => {
    const baseMemory = 768 // hidden size
    if (strat === 'standard' || strat === 'alibi' || strat === 'rope') {
      return (length * length * baseMemory) / (1024 * 1024) // MB
    } else if (strat === 'sparse') {
      return (length * Math.log2(length) * baseMemory) / (1024 * 1024)
    } else {
      // RAG
      return (length * baseMemory + 5 * 512 * baseMemory) / (1024 * 1024) // base + 检索
    }
  }

  const memoryUsage = calculateMemory(sequenceLength, selectedStrategy)

  // 可视化注意力模式
  const renderAttentionPattern = () => {
    const size = 20 // 20x20 grid
    const cells = []

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        let opacity = 0

        switch (selectedStrategy) {
          case 'standard':
            opacity = j <= i ? 0.8 : 0 // Causal mask
            break
          case 'alibi':
            // 线性衰减
            const distance = i - j
            opacity = j <= i ? Math.max(0, 0.8 - distance * 0.05) : 0
            break
          case 'rope':
            // 类似 standard 但带有旋转特性（简化显示）
            opacity = j <= i ? 0.7 : 0
            break
          case 'sparse':
            // 局部窗口 + 全局
            const isLocal = Math.abs(i - j) <= 3 && j <= i
            const isGlobal = j === 0 || i === 0
            opacity = (isLocal || isGlobal) ? 0.8 : 0
            break
          case 'rag':
            // 检索模式：特定位置高亮
            const isRetrieved = (j === 2 || j === 5 || j === 10) && j <= i
            opacity = isRetrieved ? 0.9 : (j <= i ? 0.2 : 0)
            break
        }

        cells.push(
          <div
            key={`${i}-${j}`}
            className="w-full h-full border border-gray-200"
            style={{
              backgroundColor: `rgba(${
                strategy.color === 'blue' ? '59, 130, 246' :
                strategy.color === 'green' ? '34, 197, 94' :
                strategy.color === 'purple' ? '168, 85, 247' :
                strategy.color === 'orange' ? '249, 115, 22' :
                '236, 72, 153'
              }, ${opacity})`
            }}
          />
        )
      }
    }

    return cells
  }

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">长上下文处理策略对比</h3>
        <p className="text-gray-600">探索不同方法如何突破 Transformer 的序列长度限制</p>
      </div>

      {/* 策略选择器 */}
      <div className="grid grid-cols-5 gap-2">
        {(Object.keys(strategies) as Strategy[]).map((key) => {
          const strat = strategies[key]
          return (
            <motion.button
              key={key}
              onClick={() => setSelectedStrategy(key)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedStrategy === key
                  ? `border-${strat.color}-500 bg-${strat.color}-50`
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="text-sm font-semibold">{strat.name}</div>
              <div className="text-xs text-gray-500 mt-1">{strat.complexity}</div>
            </motion.button>
          )
        })}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedStrategy}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-lg border border-gray-200 p-6"
      >
        <div className="grid grid-cols-2 gap-6">
          {/* 左侧：信息 */}
          <div className="space-y-4">
            <div>
              <div className={`text-lg font-bold text-${strategy.color}-600`}>
                {strategy.name}
              </div>
              <div className="text-sm text-gray-600 mt-1">
                {strategy.description}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500">时间复杂度</div>
                <div className="text-lg font-mono font-bold mt-1">{strategy.complexity}</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500">最大长度</div>
                <div className="text-lg font-mono font-bold mt-1">{strategy.maxLength}</div>
              </div>
            </div>

            <div>
              <div className="text-sm font-semibold text-green-700 mb-2">✓ 优势</div>
              <ul className="space-y-1">
                {strategy.advantages.map((adv, idx) => (
                  <li key={idx} className="text-sm text-gray-700 flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    {adv}
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <div className="text-sm font-semibold text-red-700 mb-2">✗ 劣势</div>
              <ul className="space-y-1">
                {strategy.disadvantages.map((dis, idx) => (
                  <li key={idx} className="text-sm text-gray-700 flex items-start">
                    <span className="text-red-500 mr-2">•</span>
                    {dis}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* 右侧：注意力模式可视化 */}
          <div className="space-y-4">
            <div>
              <div className="text-sm font-semibold mb-2">注意力模式 (20×20 示意)</div>
              <div className="aspect-square bg-gray-100 rounded-lg p-2">
                <div className="grid grid-cols-20 gap-0 h-full">
                  {renderAttentionPattern()}
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-2 text-center">
                横轴: Key tokens | 纵轴: Query tokens
              </div>
            </div>

            {/* 图例 */}
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs font-semibold mb-2">图例说明</div>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: `rgba(${
                      strategy.color === 'blue' ? '59, 130, 246' :
                      strategy.color === 'green' ? '34, 197, 94' :
                      strategy.color === 'purple' ? '168, 85, 247' :
                      strategy.color === 'orange' ? '249, 115, 22' :
                      '236, 72, 153'
                    }, 0.8)` }}
                  />
                  <span>高注意力权重</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-gray-200" />
                  <span>无注意力 / 被掩码</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 性能模拟器 */}
      <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6">
        <h4 className="text-lg font-bold mb-4">性能模拟器</h4>
        
        <div className="space-y-4">
          {/* 序列长度滑块 */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-semibold">序列长度</label>
              <span className="text-sm font-mono bg-white px-3 py-1 rounded-lg border">
                {sequenceLength} tokens
              </span>
            </div>
            <input
              type="range"
              min="256"
              max="16384"
              step="256"
              value={sequenceLength}
              onChange={(e) => setSequenceLength(Number(e.target.value))}
              className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>256</span>
              <span>4K</span>
              <span>8K</span>
              <span>16K</span>
            </div>
          </div>

          {/* 性能指标 */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-xs text-gray-500 mb-1">内存占用</div>
              <div className="text-2xl font-bold">{memoryUsage.toFixed(1)} MB</div>
              <div className="text-xs text-gray-400 mt-1">
                {memoryUsage < 100 ? '低' : memoryUsage < 500 ? '中' : '高'}
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-xs text-gray-500 mb-1">推理速度</div>
              <div className="text-2xl font-bold">
                {selectedStrategy === 'sparse' || selectedStrategy === 'rag' ? '快' :
                 sequenceLength > 4096 ? '慢' : '中'}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                相对评分
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-xs text-gray-500 mb-1">可行性</div>
              <div className="text-2xl font-bold">
                {(selectedStrategy === 'standard' && sequenceLength > 4096) ? '✗' :
                 (selectedStrategy === 'sparse' || selectedStrategy === 'rag') ? '✓' :
                 sequenceLength <= 8192 ? '✓' : '△'}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {(selectedStrategy === 'standard' && sequenceLength > 4096)
                  ? '不推荐'
                  : '可用'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 使用场景推荐 */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-bold text-blue-900 mb-2">💡 使用场景推荐</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-semibold text-blue-800">短序列 (&lt;2K)</div>
            <div className="text-blue-700">Standard Attention → 性能最优</div>
          </div>
          <div>
            <div className="font-semibold text-blue-800">中等序列 (2K-8K)</div>
            <div className="text-blue-700">RoPE + Interpolation → 平滑扩展</div>
          </div>
          <div>
            <div className="font-semibold text-blue-800">长序列 (8K-64K)</div>
            <div className="text-blue-700">Sparse Attention → 内存高效</div>
          </div>
          <div>
            <div className="font-semibold text-blue-800">超长/无限上下文</div>
            <div className="text-blue-700">RAG → 检索增强</div>
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <div className="text-xs text-gray-300 mb-2">
          {strategy.name} - 示例代码
        </div>
        <pre className="text-sm text-gray-100">
          <code>
            {selectedStrategy === 'standard' && `# 标准注意力
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
# 最大长度 ~1024 tokens`}
            {selectedStrategy === 'alibi' && `# ALiBi
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
# 训练时短，推理时可扩展`}
            {selectedStrategy === 'rope' && `# RoPE + 插值
model.config.rope_scaling = {
    "type": "linear",
    "factor": 2.0  # 扩展 2 倍
}
# 从 4K 扩展到 8K`}
            {selectedStrategy === 'sparse' && `# Sparse Attention
from transformers import LongformerModel
model = LongformerModel.from_pretrained(
    "allenai/longformer-base-4096"
)
# 局部 + 全局注意力`}
            {selectedStrategy === 'rag' && `# RAG
from transformers import RagRetriever, RagSequenceForGeneration
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever
)`}
          </code>
        </pre>
      </div>
    </div>
  )
}

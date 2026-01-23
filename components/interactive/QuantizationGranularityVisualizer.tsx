'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Layers, Grid, Box } from 'lucide-react'

type Granularity = 'tensor' | 'channel' | 'group'

export default function QuantizationGranularityVisualizer() {
  const [granularity, setGranularity] = useState<Granularity>('tensor')
  const [showFormula, setShowFormula] = useState(false)

  // 模拟权重矩阵 (4x8)
  const weights = Array.from({ length: 4 }, (_, i) =>
    Array.from({ length: 8 }, (_, j) => 
      Math.sin(i * 0.5 + j * 0.3) * 2 + Math.random() * 0.5
    )
  )

  const getQuantizationInfo = () => {
    switch (granularity) {
      case 'tensor':
        const globalMin = Math.min(...weights.flat())
        const globalMax = Math.max(...weights.flat())
        const globalScale = (globalMax - globalMin) / 255
        return {
          scales: [globalScale],
          groups: weights.map(() => Array(8).fill(0)),
          description: '整个张量共享 1 个 scale',
          formula: 's = \\frac{\\max(\\mathbf{W}) - \\min(\\mathbf{W})}{2^b - 1}',
          scaleCount: 1,
        }
      
      case 'channel':
        const channelScales = weights.map(channel => {
          const min = Math.min(...channel)
          const max = Math.max(...channel)
          return (max - min) / 255
        })
        return {
          scales: channelScales,
          groups: weights.map((_, channelIdx) => 
            Array(8).fill(channelIdx)
          ),
          description: `每个输出通道独立量化 (${weights.length} 个 scale)`,
          formula: 's_i = \\frac{\\max(\\mathbf{W}_i) - \\min(\\mathbf{W}_i)}{2^b - 1}',
          scaleCount: weights.length,
        }
      
      case 'group':
        const groupSize = 4
        const numGroups = Math.ceil(weights[0].length / groupSize)
        const groupScales: number[] = []
        const groupAssignments: number[][] = []
        
        weights.forEach((channel, channelIdx) => {
          const channelGroups: number[] = []
          for (let g = 0; g < numGroups; g++) {
            const start = g * groupSize
            const end = Math.min(start + groupSize, channel.length)
            const groupValues = channel.slice(start, end)
            const min = Math.min(...groupValues)
            const max = Math.max(...groupValues)
            groupScales.push((max - min) / 255)
            
            for (let i = start; i < end; i++) {
              channelGroups.push(channelIdx * numGroups + g)
            }
          }
          groupAssignments.push(channelGroups)
        })
        
        return {
          scales: groupScales,
          groups: groupAssignments,
          description: `每 ${groupSize} 列为一组 (${groupScales.length} 个 scale)`,
          formula: 's_{g} = \\frac{\\max(\\mathbf{W}_{[:, g \\cdot k:(g+1) \\cdot k]}) - \\min(\\mathbf{W}_{[:, g \\cdot k:(g+1) \\cdot k]})}{2^b - 1}',
          scaleCount: groupScales.length,
        }
    }
  }

  const info = getQuantizationInfo()
  const colorMap = [
    'from-red-400 to-red-600',
    'from-blue-400 to-blue-600',
    'from-green-400 to-green-600',
    'from-yellow-400 to-yellow-600',
    'from-purple-400 to-purple-600',
    'from-pink-400 to-pink-600',
    'from-indigo-400 to-indigo-600',
    'from-cyan-400 to-cyan-600',
  ]

  const options = [
    { value: 'tensor' as const, label: 'Per-Tensor', icon: Box, desc: '张量级' },
    { value: 'channel' as const, label: 'Per-Channel', icon: Layers, desc: '通道级' },
    { value: 'group' as const, label: 'Per-Group', icon: Grid, desc: '分组级' },
  ]

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        量化粒度可视化
      </h3>

      {/* 粒度选择 */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {options.map((option) => {
          const Icon = option.icon
          return (
            <motion.button
              key={option.value}
              onClick={() => setGranularity(option.value)}
              className={`p-4 rounded-lg border-2 transition-all ${
                granularity === option.value
                  ? 'border-purple-500 bg-white shadow-lg'
                  : 'border-slate-300 bg-white/50 hover:border-purple-300'
              }`}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <Icon className={`w-6 h-6 mx-auto mb-2 ${
                granularity === option.value ? 'text-purple-600' : 'text-slate-400'
              }`} />
              <div className={`font-bold text-sm ${
                granularity === option.value ? 'text-purple-600' : 'text-slate-600'
              }`}>
                {option.label}
              </div>
              <div className="text-xs text-slate-500 mt-1">{option.desc}</div>
            </motion.button>
          )
        })}
      </div>

      {/* 权重矩阵可视化 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-slate-800">权重矩阵分组</h4>
          <div className="text-sm text-slate-600">
            {info.scaleCount} 个 scale 参数
          </div>
        </div>
        
        <div className="grid gap-1 mb-4">
          {weights.map((channel, channelIdx) => (
            <div key={channelIdx} className="flex gap-1">
              {channel.map((weight, colIdx) => {
                const groupId = info.groups[channelIdx][colIdx]
                const color = colorMap[groupId % colorMap.length]
                return (
                  <motion.div
                    key={colIdx}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: (channelIdx * 8 + colIdx) * 0.01 }}
                    className={`flex-1 h-12 rounded bg-gradient-to-br ${color} flex items-center justify-center text-white text-xs font-mono relative group`}
                  >
                    {weight.toFixed(1)}
                    <div className="absolute bottom-full mb-2 hidden group-hover:block bg-slate-800 text-white px-2 py-1 rounded text-xs whitespace-nowrap z-10">
                      Group {groupId} | Scale {info.scales[groupId].toFixed(4)}
                    </div>
                  </motion.div>
                )
              })}
            </div>
          ))}
        </div>

        <div className="text-sm text-slate-600 bg-slate-50 p-3 rounded-lg">
          {info.description}
        </div>
      </div>

      {/* 优缺点对比 */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h5 className="font-bold text-green-800 mb-2">✅ 优点</h5>
          <ul className="text-sm text-green-700 space-y-1">
            {granularity === 'tensor' && (
              <>
                <li>• 内存高效 (仅 1 个 scale)</li>
                <li>• 计算简单</li>
                <li>• 硬件友好</li>
              </>
            )}
            {granularity === 'channel' && (
              <>
                <li>• 精度更高 (适应每通道分布)</li>
                <li>• 常用于 CNN/Transformer</li>
                <li>• 硬件支持良好</li>
              </>
            )}
            {granularity === 'group' && (
              <>
                <li>• 平衡精度与内存</li>
                <li>• GPTQ/AWQ 默认方式</li>
                <li>• 灵活调整 group_size</li>
              </>
            )}
          </ul>
        </div>

        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
          <h5 className="font-bold text-red-800 mb-2">❌ 缺点</h5>
          <ul className="text-sm text-red-700 space-y-1">
            {granularity === 'tensor' && (
              <>
                <li>• 精度较低</li>
                <li>• 无法适应异质分布</li>
                <li>• 大模型效果差</li>
              </>
            )}
            {granularity === 'channel' && (
              <>
                <li>• 额外存储 C 个 scale</li>
                <li>• 对小模型开销较大</li>
              </>
            )}
            {granularity === 'group' && (
              <>
                <li>• 需调优 group_size</li>
                <li>• 内存占用介于中间</li>
                <li>• 实现稍复杂</li>
              </>
            )}
          </ul>
        </div>
      </div>

      {/* 数学公式 */}
      <motion.button
        onClick={() => setShowFormula(!showFormula)}
        className="w-full p-4 bg-purple-100 rounded-lg border border-purple-300 hover:bg-purple-200 transition-colors"
      >
        <div className="font-bold text-purple-800">
          {showFormula ? '隐藏' : '显示'} 量化公式
        </div>
      </motion.button>

      <AnimatePresence>
        {showFormula && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 bg-white p-6 rounded-xl border border-slate-200"
          >
            <div className="font-mono text-sm text-center mb-3 overflow-x-auto">
              ${info.formula}$
            </div>
            <div className="text-sm text-slate-600">
              <strong>量化公式：</strong>
              <div className="mt-2 font-mono text-xs bg-slate-50 p-3 rounded">
                $\mathbf{'{'}W{'}'}_q = \text{'{'}round{'}'}(\mathbf{'{'}W{'}'} / s) \cdot s$
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 悬停在权重方块上查看分组信息 | 相同颜色 = 共享 scale
      </div>
    </div>
  )
}

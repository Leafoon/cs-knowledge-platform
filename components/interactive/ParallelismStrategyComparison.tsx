'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Database, Layers, GitBranch, CheckCircle } from 'lucide-react'

type StrategyType = 'data' | 'model' | 'hybrid'

interface Strategy {
  type: StrategyType
  name: string
  icon: React.ReactNode
  description: string
  pros: string[]
  cons: string[]
  useCase: string
  memoryFormula: string
  communication: string
}

const strategies: Strategy[] = [
  {
    type: 'data',
    name: '数据并行 (Data Parallelism)',
    icon: <Database className="w-6 h-6" />,
    description: '每个GPU持有完整模型副本，数据切分到不同GPU',
    pros: [
      '实现简单，易于调试',
      '通信开销相对较小',
      '支持动态batch size',
      '负载均衡性好'
    ],
    cons: [
      '每个GPU需要完整模型',
      '显存占用 = 模型大小 × GPU数',
      '大模型无法使用',
      '梯度同步成为瓶颈'
    ],
    useCase: '小模型 (<10B)，多GPU训练加速',
    memoryFormula: 'Memory_per_GPU = 16 × M',
    communication: 'AllReduce 梯度 (2×M per iteration)'
  },
  {
    type: 'model',
    name: '模型并行 (Model Parallelism)',
    icon: <Layers className="w-6 h-6" />,
    description: '模型切分到不同GPU，每个GPU只持有部分层',
    pros: [
      '可训练超大模型',
      '显存占用 = 模型大小 / GPU数',
      '突破单卡显存限制',
      '支持更长序列'
    ],
    cons: [
      'GPU利用率低（流水线气泡）',
      '层间通信频繁',
      '实现复杂',
      '负载不均衡'
    ],
    useCase: '超大模型 (100B+)，单卡显存不足',
    memoryFormula: 'Memory_per_GPU = 16 × M / N',
    communication: 'P2P 激活值传递 (频繁)'
  },
  {
    type: 'hybrid',
    name: '混合并行 (Hybrid Parallelism)',
    icon: <GitBranch className="w-6 h-6" />,
    description: '数据并行 + 模型并行 + Pipeline并行',
    pros: [
      '兼顾效率与可扩展性',
      '灵活调整并行维度',
      '最大化GPU利用率',
      '支持千亿级模型'
    ],
    cons: [
      '配置复杂',
      '调试困难',
      '需要专业知识',
      '通信拓扑复杂'
    ],
    useCase: '超大规模模型 (175B+)，多节点训练',
    memoryFormula: 'Memory = 16×M / (DP×PP×TP)',
    communication: 'AllReduce + P2P + Broadcast'
  }
]

export default function ParallelismStrategyComparison() {
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyType>('data')
  
  const currentStrategy = strategies.find(s => s.type === selectedStrategy)!

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <GitBranch className="w-8 h-8 text-purple-600" />
        <h3 className="text-2xl font-bold text-slate-800">分布式训练并行策略对比</h3>
      </div>

      {/* 策略选择器 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {strategies.map((strategy) => (
          <motion.button
            key={strategy.type}
            onClick={() => setSelectedStrategy(strategy.type)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedStrategy === strategy.type
                ? 'border-purple-600 bg-purple-50 shadow-lg'
                : 'border-slate-200 bg-white hover:border-purple-300'
            }`}
          >
            <div className="flex items-center gap-3 mb-2">
              <div className={`${
                selectedStrategy === strategy.type ? 'text-purple-600' : 'text-slate-500'
              }`}>
                {strategy.icon}
              </div>
              <div className={`font-bold text-left ${
                selectedStrategy === strategy.type ? 'text-purple-900' : 'text-slate-700'
              }`}>
                {strategy.name.split('(')[0]}
              </div>
            </div>
            <div className="text-xs text-slate-500 text-left">
              {strategy.description}
            </div>
          </motion.button>
        ))}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedStrategy}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-4"
      >
        {/* 优缺点对比 */}
        <div className="grid grid-cols-2 gap-4">
          {/* 优点 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <h4 className="font-bold text-green-800">优点</h4>
            </div>
            <ul className="space-y-2">
              {currentStrategy.pros.map((pro, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-700"
                >
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>{pro}</span>
                </motion.li>
              ))}
            </ul>
          </div>

          {/* 缺点 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-5 h-5 text-red-600" />
              <h4 className="font-bold text-red-800">缺点</h4>
            </div>
            <ul className="space-y-2">
              {currentStrategy.cons.map((con, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-700"
                >
                  <span className="text-red-600 mt-0.5">✗</span>
                  <span>{con}</span>
                </motion.li>
              ))}
            </ul>
          </div>
        </div>

        {/* 技术细节 */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg shadow">
            <div className="text-sm font-medium text-blue-700 mb-2">显存占用公式</div>
            <div className="font-mono text-sm text-blue-900 bg-white p-2 rounded">
              {currentStrategy.memoryFormula}
            </div>
            <div className="text-xs text-blue-600 mt-2">
              M = 模型参数量, N = GPU数
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg shadow">
            <div className="text-sm font-medium text-purple-700 mb-2">通信模式</div>
            <div className="font-mono text-sm text-purple-900 bg-white p-2 rounded">
              {currentStrategy.communication}
            </div>
            <div className="text-xs text-purple-600 mt-2">
              每次迭代通信量
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg shadow">
            <div className="text-sm font-medium text-green-700 mb-2">适用场景</div>
            <div className="text-sm text-green-900 bg-white p-2 rounded">
              {currentStrategy.useCase}
            </div>
            <div className="text-xs text-green-600 mt-2">
              最佳实践
            </div>
          </div>
        </div>

        {/* 可视化示意图 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-4">架构示意图</h4>
          
          {selectedStrategy === 'data' && (
            <div className="flex justify-center items-center gap-8">
              {[0, 1, 2, 3].map((idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex flex-col items-center"
                >
                  <div className="w-20 h-24 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg shadow-lg flex items-center justify-center text-white font-bold mb-2">
                    GPU {idx}
                  </div>
                  <div className="text-xs text-slate-600 text-center">
                    完整模型<br/>
                    数据分片 {idx + 1}/4
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {selectedStrategy === 'model' && (
            <div className="flex justify-center items-center gap-4">
              {['Layer 1-8', 'Layer 9-16', 'Layer 17-24', 'Layer 25-32'].map((label, idx) => (
                <React.Fragment key={idx}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.15 }}
                    className="flex flex-col items-center"
                  >
                    <div className="w-24 h-32 bg-gradient-to-br from-purple-400 to-purple-600 rounded-lg shadow-lg flex items-center justify-center text-white font-bold text-sm mb-2">
                      GPU {idx}
                    </div>
                    <div className="text-xs text-slate-600 text-center">
                      {label}
                    </div>
                  </motion.div>
                  {idx < 3 && (
                    <div className="text-2xl text-slate-400">→</div>
                  )}
                </React.Fragment>
              ))}
            </div>
          )}

          {selectedStrategy === 'hybrid' && (
            <div className="space-y-4">
              <div className="text-center text-sm text-slate-600 mb-2">
                数据并行 (DP=2) × Pipeline并行 (PP=2) × 张量并行 (TP=2) = 8 GPUs
              </div>
              <div className="grid grid-cols-4 gap-3">
                {Array.from({ length: 8 }).map((_, idx) => {
                  const dp = Math.floor(idx / 4)
                  const pp = Math.floor((idx % 4) / 2)
                  const tp = idx % 2
                  return (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: idx * 0.08 }}
                      className="p-3 bg-gradient-to-br from-green-400 to-green-600 rounded-lg shadow text-white text-center"
                    >
                      <div className="font-bold">GPU {idx}</div>
                      <div className="text-xs mt-1 space-y-0.5">
                        <div>DP: {dp}</div>
                        <div>PP: {pp}</div>
                        <div>TP: {tp}</div>
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}

'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, TrendingDown, Zap, Database } from 'lucide-react'

type ZeROStage = 'ddp' | 'zero1' | 'zero2' | 'zero3'

interface StageConfig {
  id: ZeROStage
  name: string
  description: string
  shardOptimizer: boolean
  shardGradient: boolean
  shardParameter: boolean
  memoryPerGPU: number
  commOverhead: string
  speedRelative: number
}

const stages: StageConfig[] = [
  {
    id: 'ddp',
    name: 'DDP (Baseline)',
    description: '标准数据并行，每GPU持有完整副本',
    shardOptimizer: false,
    shardGradient: false,
    shardParameter: false,
    memoryPerGPU: 112, // 7B model: 16 × 7GB
    commOverhead: '1.0×',
    speedRelative: 100,
  },
  {
    id: 'zero1',
    name: 'ZeRO-1',
    description: '分片优化器状态',
    shardOptimizer: true,
    shardGradient: false,
    shardParameter: false,
    memoryPerGPU: 31, // (2+2+12/4) × 7GB
    commOverhead: '1.0×',
    speedRelative: 100,
  },
  {
    id: 'zero2',
    name: 'ZeRO-2',
    description: '分片优化器状态 + 梯度',
    shardOptimizer: true,
    shardGradient: true,
    shardParameter: false,
    memoryPerGPU: 17.5, // (2+2/4+12/4) × 7GB
    commOverhead: '1.5×',
    speedRelative: 95,
  },
  {
    id: 'zero3',
    name: 'ZeRO-3',
    description: '分片所有模型状态（参数+梯度+优化器）',
    shardOptimizer: true,
    shardGradient: true,
    shardParameter: true,
    memoryPerGPU: 7, // 16/4 × 7GB
    commOverhead: '2.0×',
    speedRelative: 85,
  },
]

export default function DeepSpeedZeROStages() {
  const [selectedStage, setSelectedStage] = useState<ZeROStage>('zero2')
  
  const currentStage = stages.find(s => s.id === selectedStage)!
  const baselineStage = stages[0] // DDP
  const gpuCount = 4

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-emerald-600" />
        <h3 className="text-2xl font-bold text-slate-800">DeepSpeed ZeRO 三阶段对比</h3>
      </div>

      {/* 阶段选择器 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {stages.map((stage) => (
          <button
            key={stage.id}
            onClick={() => setSelectedStage(stage.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedStage === stage.id
                ? 'border-emerald-600 bg-emerald-50 shadow-lg'
                : 'border-slate-200 bg-white hover:border-emerald-300'
            }`}
          >
            <div className={`font-bold mb-2 ${
              selectedStage === stage.id ? 'text-emerald-900' : 'text-slate-700'
            }`}>
              {stage.name}
            </div>
            <div className="text-xs text-slate-600 mb-3">
              {stage.description}
            </div>
            <div className="text-2xl font-bold text-emerald-600">
              {stage.memoryPerGPU} GB
            </div>
            <div className="text-xs text-slate-500">每GPU显存</div>
          </button>
        ))}
      </div>

      {/* 分片策略可视化 */}
      <div className="mb-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">分片策略</h4>
        
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: '优化器状态', sharded: currentStage.shardOptimizer, size: 12, color: 'purple' },
            { label: '梯度', sharded: currentStage.shardGradient, size: 2, color: 'green' },
            { label: '模型参数', sharded: currentStage.shardParameter, size: 2, color: 'blue' },
          ].map((item, idx) => (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-700">{item.label}</span>
                <span className={`px-2 py-1 rounded text-xs font-bold ${
                  item.sharded
                    ? 'bg-green-100 text-green-700'
                    : 'bg-slate-100 text-slate-600'
                }`}>
                  {item.sharded ? '✓ 分片' : '✗ 复制'}
                </span>
              </div>

              {/* GPU显存占用 */}
              <div className="grid grid-cols-4 gap-1">
                {Array.from({ length: gpuCount }).map((_, gpuId) => {
                  const memorySize = item.sharded ? item.size / gpuCount : item.size
                  
                  return (
                    <motion.div
                      key={gpuId}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: idx * 0.1 + gpuId * 0.05 }}
                      className={`h-16 rounded flex flex-col items-center justify-center text-white text-xs font-bold bg-gradient-to-br ${
                        item.color === 'purple'
                          ? 'from-purple-400 to-purple-600'
                          : item.color === 'green'
                          ? 'from-green-400 to-green-600'
                          : 'from-blue-400 to-blue-600'
                      } ${!item.sharded ? 'opacity-100' : 'opacity-80'}`}
                    >
                      <div>GPU {gpuId}</div>
                      <div className="text-xs">{memorySize} GB</div>
                    </motion.div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 性能对比 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* 显存节省 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-5 h-5 text-green-600" />
            <h4 className="font-bold text-slate-800">显存节省</h4>
          </div>
          
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600">基线 (DDP)</span>
                <span className="font-mono font-bold">{baselineStage.memoryPerGPU} GB</span>
              </div>
              <div className="h-6 bg-slate-200 rounded" />
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600">{currentStage.name}</span>
                <span className="font-mono font-bold text-green-600">
                  {currentStage.memoryPerGPU} GB
                </span>
              </div>
              <div className="relative h-6 bg-slate-100 rounded overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(currentStage.memoryPerGPU / baselineStage.memoryPerGPU) * 100}%` }}
                  transition={{ duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-green-400 to-green-600"
                />
              </div>
            </div>

            <div className="pt-2 border-t border-slate-200">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {((1 - currentStage.memoryPerGPU / baselineStage.memoryPerGPU) * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500">节省显存</div>
              </div>
            </div>
          </div>
        </div>

        {/* 通信开销 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-yellow-600" />
            <h4 className="font-bold text-slate-800">通信开销</h4>
          </div>

          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 mb-2">相对DDP</div>
              <div className="text-4xl font-bold text-yellow-600 text-center">
                {currentStage.commOverhead}
              </div>
            </div>

            <div className="p-3 bg-yellow-50 rounded border border-yellow-200">
              <div className="text-xs text-slate-700">
                {currentStage.id === 'ddp' && '仅梯度AllReduce'}
                {currentStage.id === 'zero1' && '与DDP相同'}
                {currentStage.id === 'zero2' && '增加梯度Reduce-Scatter'}
                {currentStage.id === 'zero3' && '额外AllGather参数分片'}
              </div>
            </div>
          </div>
        </div>

        {/* 训练速度 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-5 h-5 text-blue-600" />
            <h4 className="font-bold text-slate-800">训练速度</h4>
          </div>

          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 mb-2">相对性能</div>
              <div className="text-4xl font-bold text-blue-600 text-center">
                {currentStage.speedRelative}%
              </div>
            </div>

            <div className="relative h-8 bg-slate-100 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${currentStage.speedRelative}%` }}
                transition={{ duration: 0.8 }}
                className="h-full bg-gradient-to-r from-blue-400 to-blue-600 flex items-center justify-end pr-2"
              >
                <span className="text-white text-xs font-bold">
                  {currentStage.speedRelative}%
                </span>
              </motion.div>
            </div>

            <div className="text-xs text-slate-600">
              {currentStage.speedRelative === 100
                ? '与基线相同'
                : `约慢 ${100 - currentStage.speedRelative}%（通信开销）`}
            </div>
          </div>
        </div>
      </div>

      {/* 使用建议 */}
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 p-5 rounded-lg shadow border border-emerald-200">
        <h4 className="font-bold text-slate-800 mb-3">选择建议</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-bold text-emerald-800 mb-2">模型规模</div>
            <ul className="space-y-1 text-slate-700">
              <li><span className="font-mono bg-white px-2 py-0.5 rounded">&lt;10B</span>: DDP</li>
              <li><span className="font-mono bg-white px-2 py-0.5 rounded">10B-30B</span>: ZeRO-2</li>
              <li><span className="font-mono bg-white px-2 py-0.5 rounded">30B-70B</span>: ZeRO-3</li>
              <li><span className="font-mono bg-white px-2 py-0.5 rounded">&gt;70B</span>: ZeRO-3 + Offload</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-emerald-800 mb-2">权衡考虑</div>
            <ul className="space-y-1 text-slate-700">
              <li>✓ 显存不足 → 选择更高阶段</li>
              <li>✓ 追求速度 → 选择较低阶段</li>
              <li>✓ 通信慢 → 避免ZeRO-3</li>
              <li>✓ 超大模型 → 必须ZeRO-3</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

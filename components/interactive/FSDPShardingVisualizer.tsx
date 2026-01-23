'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Layers, ArrowRightLeft, Database, Zap } from 'lucide-react'

type Phase = 'init' | 'allgather' | 'compute' | 'reducescatter'

const phases: { id: Phase; name: string; description: string }[] = [
  { id: 'init', name: '初始状态', description: '参数分片存储在各GPU' },
  { id: 'allgather', name: 'AllGather', description: '收集分片参数，重建完整层' },
  { id: 'compute', name: '计算', description: '前向/反向传播' },
  { id: 'reducescatter', name: 'ReduceScatter', description: '梯度归约并重新分片' },
]

export default function FSDPShardingVisualizer() {
  const [currentPhase, setCurrentPhase] = useState<Phase>('init')
  const [hoveredGpu, setHoveredGpu] = useState<number | null>(null)

  const gpuCount = 4
  const totalParams = 16 // 模拟16个参数块

  // 获取每个GPU在当前阶段持有的参数
  const getGpuParams = (gpuId: number) => {
    const shardSize = totalParams / gpuCount
    const ownShard = Array.from(
      { length: shardSize },
      (_, i) => gpuId * shardSize + i
    )

    switch (currentPhase) {
      case 'init':
      case 'reducescatter':
        return ownShard // 只持有自己的分片
      case 'allgather':
      case 'compute':
        return Array.from({ length: totalParams }, (_, i) => i) // 持有完整参数
      default:
        return ownShard
    }
  }

  // 获取内存占用
  const getMemoryUsage = () => {
    switch (currentPhase) {
      case 'init':
      case 'reducescatter':
        return 25 // 25%（1/4参数）
      case 'allgather':
      case 'compute':
        return 100 // 100%（完整参数）
      default:
        return 25
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-cyan-600" />
        <h3 className="text-2xl font-bold text-slate-800">FSDP 参数分片可视化</h3>
      </div>

      {/* 阶段选择器 */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {phases.map((phase, idx) => (
          <button
            key={phase.id}
            onClick={() => setCurrentPhase(phase.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              currentPhase === phase.id
                ? 'border-cyan-600 bg-cyan-50 shadow-md'
                : 'border-slate-200 bg-white hover:border-cyan-300'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 rounded-full bg-cyan-600 text-white flex items-center justify-center text-sm font-bold">
                {idx + 1}
              </div>
              <div className={`font-bold ${
                currentPhase === phase.id ? 'text-cyan-900' : 'text-slate-700'
              }`}>
                {phase.name}
              </div>
            </div>
            <div className="text-xs text-slate-600">{phase.description}</div>
          </button>
        ))}
      </div>

      {/* GPU可视化 */}
      <div className="mb-6">
        <div className="grid grid-cols-4 gap-4">
          {Array.from({ length: gpuCount }).map((_, gpuId) => {
            const params = getGpuParams(gpuId)
            const shardSize = totalParams / gpuCount

            return (
              <motion.div
                key={gpuId}
                onMouseEnter={() => setHoveredGpu(gpuId)}
                onMouseLeave={() => setHoveredGpu(null)}
                className="bg-white p-4 rounded-lg shadow-lg"
              >
                {/* GPU标题 */}
                <div className="bg-gradient-to-br from-cyan-400 to-cyan-600 text-white rounded-lg p-3 mb-3 text-center">
                  <div className="font-bold text-lg">GPU {gpuId}</div>
                  <div className="text-xs opacity-90">
                    {currentPhase === 'init' || currentPhase === 'reducescatter'
                      ? `Shard ${gpuId}`
                      : 'Full Model'}
                  </div>
                </div>

                {/* 参数块网格 */}
                <div className="grid grid-cols-4 gap-1 mb-3">
                  {Array.from({ length: totalParams }).map((_, paramId) => {
                    const hasParam = params.includes(paramId)
                    const isOwnShard = paramId >= gpuId * shardSize && paramId < (gpuId + 1) * shardSize

                    return (
                      <motion.div
                        key={paramId}
                        layout
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{
                          opacity: hasParam ? 1 : 0.2,
                          scale: hasParam ? 1 : 0.7,
                        }}
                        transition={{ duration: 0.3 }}
                        className={`aspect-square rounded ${
                          hasParam
                            ? isOwnShard
                              ? 'bg-cyan-500'
                              : 'bg-purple-400'
                            : 'bg-slate-200'
                        }`}
                        title={`Param ${paramId}`}
                      />
                    )
                  })}
                </div>

                {/* 显存占用 */}
                <div className="text-xs text-slate-600 mb-1">显存占用</div>
                <div className="h-6 bg-slate-100 rounded overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${getMemoryUsage()}%` }}
                    transition={{ duration: 0.5 }}
                    className={`h-full ${
                      getMemoryUsage() === 100
                        ? 'bg-red-500'
                        : 'bg-green-500'
                    } flex items-center justify-center text-white text-xs font-bold`}
                  >
                    {getMemoryUsage()}%
                  </motion.div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* 通信示意 */}
      <AnimatePresence mode="wait">
        {(currentPhase === 'allgather' || currentPhase === 'reducescatter') && (
          <motion.div
            key={currentPhase}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6 p-4 bg-white rounded-lg shadow"
          >
            <div className="flex items-center gap-3 mb-3">
              <ArrowRightLeft className="w-6 h-6 text-cyan-600" />
              <h4 className="font-bold text-slate-800">
                {currentPhase === 'allgather' ? 'AllGather 通信' : 'ReduceScatter 通信'}
              </h4>
            </div>
            <div className="text-sm text-slate-700">
              {currentPhase === 'allgather' ? (
                <div>
                  每个GPU从其他GPU收集参数分片，组装成完整模型。
                  <span className="font-mono bg-slate-100 px-2 py-1 rounded ml-2">
                    通信量 = 3 × M / 4
                  </span>
                </div>
              ) : (
                <div>
                  梯度在所有GPU间归约求和，然后每个GPU只保留自己的分片。
                  <span className="font-mono bg-slate-100 px-2 py-1 rounded ml-2">
                    通信量 = M / 4
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 阶段详情 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-5 h-5 text-blue-600" />
            <h4 className="font-bold text-slate-800">显存优化</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-600">DDP (每GPU完整模型)</span>
              <span className="font-mono font-bold text-red-600">100%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">FSDP (参数分片)</span>
              <span className="font-mono font-bold text-green-600">25%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">节省</span>
              <span className="font-mono font-bold text-blue-600">75%</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-lg shadow">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-yellow-600" />
            <h4 className="font-bold text-slate-800">通信开销</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-600">AllGather (前向)</span>
              <span className="font-mono text-slate-700">0.75M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">ReduceScatter (反向)</span>
              <span className="font-mono text-slate-700">0.25M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">总计/迭代</span>
              <span className="font-mono font-bold text-yellow-600">1M</span>
            </div>
          </div>
        </div>
      </div>

      {/* 工作流程总结 */}
      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 p-5 rounded-lg shadow border border-cyan-200">
        <h4 className="font-bold text-slate-800 mb-3">FSDP 工作流程</h4>
        <div className="grid grid-cols-4 gap-2 text-center text-sm">
          <div className="bg-white p-3 rounded border-2 border-slate-300">
            <div className="font-bold text-slate-700 mb-1">1. 分片存储</div>
            <div className="text-xs text-slate-600">参数分布到4个GPU</div>
          </div>
          <div className="bg-white p-3 rounded border-2 border-cyan-400">
            <div className="font-bold text-cyan-700 mb-1">2. AllGather</div>
            <div className="text-xs text-slate-600">收集完整层参数</div>
          </div>
          <div className="bg-white p-3 rounded border-2 border-green-400">
            <div className="font-bold text-green-700 mb-1">3. 计算</div>
            <div className="text-xs text-slate-600">前向/反向传播</div>
          </div>
          <div className="bg-white p-3 rounded border-2 border-purple-400">
            <div className="font-bold text-purple-700 mb-1">4. ReduceScatter</div>
            <div className="text-xs text-slate-600">梯度归约并分片</div>
          </div>
        </div>
      </div>
    </div>
  )
}

'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, RotateCcw, GitBranch } from 'lucide-react'

type PipelineMode = 'naive' | 'gpipe'

interface MicroBatch {
  id: number
  stage: number
  phase: 'forward' | 'backward'
  color: string
}

export default function PipelineParallelismVisualizer() {
  const [mode, setMode] = useState<PipelineMode>('naive')
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const stages = 4
  const microBatches = mode === 'naive' ? 4 : 8

  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(() => {
        setStep((s) => (s + 1) % (microBatches * 2))
      }, 600)
      return () => clearTimeout(timer)
    }
  }, [isPlaying, step, microBatches])

  const reset = () => {
    setStep(0)
    setIsPlaying(false)
  }

  // 获取当前时间步每个stage的任务
  const getStageTask = (stage: number): MicroBatch | null => {
    if (mode === 'naive') {
      // 朴素pipeline: 一次处理一个batch
      const batchId = Math.floor(step / 2)
      const isForward = step % 2 === 0
      
      if (isForward) {
        if (step === stage) return { id: batchId, stage, phase: 'forward', color: 'blue' }
      } else {
        if (step === stages + (stages - 1 - stage)) {
          return { id: batchId, stage, phase: 'backward', color: 'purple' }
        }
      }
      return null
    } else {
      // GPipe: 微批次流水线
      const colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'cyan', 'yellow']
      
      // 前向阶段
      if (step < microBatches + stages - 1) {
        const batchId = step - stage
        if (batchId >= 0 && batchId < microBatches) {
          return { id: batchId, stage, phase: 'forward', color: colors[batchId] }
        }
      }
      
      // 反向阶段
      if (step >= microBatches) {
        const backwardStart = microBatches + stages - 1
        const batchId = step - backwardStart - (stages - 1 - stage)
        if (batchId >= 0 && batchId < microBatches) {
          return { id: batchId, stage, phase: 'backward', color: colors[batchId] }
        }
      }
      
      return null
    }
  }

  // 计算气泡率
  const getBubbleRate = () => {
    if (mode === 'naive') {
      return ((stages - 1) / stages * 100).toFixed(0)
    } else {
      const totalSteps = microBatches + 2 * (stages - 1)
      const bubbleSteps = 2 * (stages - 1)
      return ((bubbleSteps / totalSteps) * 100).toFixed(0)
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <GitBranch className="w-8 h-8 text-violet-600" />
          <h3 className="text-2xl font-bold text-slate-800">Pipeline 并行可视化</h3>
        </div>

        {/* 控制按钮 */}
        <div className="flex items-center gap-3">
          <div className="flex gap-2">
            <button
              onClick={() => { setMode('naive'); reset() }}
              className={`px-4 py-2 rounded-lg transition-colors ${
                mode === 'naive'
                  ? 'bg-violet-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
              }`}
            >
              朴素Pipeline
            </button>
            <button
              onClick={() => { setMode('gpipe'); reset() }}
              className={`px-4 py-2 rounded-lg transition-colors ${
                mode === 'gpipe'
                  ? 'bg-violet-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
              }`}
            >
              GPipe (微批次)
            </button>
          </div>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors flex items-center gap-2"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={reset}
            className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">Pipeline阶段数</div>
          <div className="text-3xl font-bold text-violet-600">{stages}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">微批次数</div>
          <div className="text-3xl font-bold text-green-600">{microBatches}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">气泡率</div>
          <div className="text-3xl font-bold text-red-600">{getBubbleRate()}%</div>
        </div>
      </div>

      {/* Pipeline可视化 */}
      <div className="bg-white p-6 rounded-lg shadow mb-6">
        <div className="space-y-3">
          {Array.from({ length: stages }).map((_, stage) => (
            <div key={stage} className="flex items-center gap-3">
              <div className="w-24 font-bold text-slate-700">
                GPU {stage}
              </div>
              
              <div className="flex-1 grid grid-cols-16 gap-1">
                {Array.from({ length: microBatches * 2 }).map((_, timeStep) => {
                  const currentStep = step
                  const task = timeStep === currentStep ? getStageTask(stage) : null
                  const isBubble = timeStep <= currentStep && !task && timeStep < currentStep
                  const isActive = timeStep === currentStep && task
                  
                  return (
                    <motion.div
                      key={timeStep}
                      initial={{ opacity: 0.3 }}
                      animate={{
                        opacity: isActive ? 1 : isBubble ? 0.3 : 0.15,
                        scale: isActive ? 1.1 : 1,
                      }}
                      className={`h-12 rounded flex items-center justify-center text-xs font-bold ${
                        task
                          ? task.phase === 'forward'
                            ? `bg-gradient-to-br from-${task.color}-400 to-${task.color}-600 text-white`
                            : `bg-gradient-to-br from-purple-400 to-purple-600 text-white`
                          : isBubble
                          ? 'bg-slate-300'
                          : 'bg-slate-100'
                      }`}
                    >
                      {task && (
                        <div className="text-center">
                          <div>{task.phase === 'forward' ? 'F' : 'B'}</div>
                          <div className="text-[10px]">{task.id}</div>
                        </div>
                      )}
                    </motion.div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* 图例 */}
        <div className="mt-4 flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-blue-400 to-blue-600 rounded" />
            <span className="text-slate-700">前向传播 (F)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-purple-400 to-purple-600 rounded" />
            <span className="text-slate-700">反向传播 (B)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-slate-300 rounded" />
            <span className="text-slate-700">气泡 (Bubble)</span>
          </div>
        </div>
      </div>

      {/* 对比说明 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-red-50 to-orange-50 p-5 rounded-lg shadow border border-red-200">
          <h4 className="font-bold text-red-800 mb-3">朴素Pipeline</h4>
          <ul className="space-y-2 text-sm text-slate-700">
            <li>✗ 气泡率高 (75%)</li>
            <li>✗ GPU利用率低</li>
            <li>✓ 实现简单</li>
            <li>✗ 不适合生产环境</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-5 rounded-lg shadow border border-green-200">
          <h4 className="font-bold text-green-800 mb-3">GPipe (微批次)</h4>
          <ul className="space-y-2 text-sm text-slate-700">
            <li>✓ 气泡率降低 (~{getBubbleRate()}%)</li>
            <li>✓ GPU利用率提升</li>
            <li>✓ 支持超大模型</li>
            <li>✓ 生产级方案</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

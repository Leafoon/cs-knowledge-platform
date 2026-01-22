'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Clock, MemoryStick, Cpu } from 'lucide-react'

interface SubStep {
  name: string
  time: number
  memory: number
  description: string
}

interface Step {
  id: string
  name: string
  substeps: SubStep[]
  totalTime: number
  color: string
}

export default function TrainingStepBreakdown() {
  const [activeStep, setActiveStep] = useState(0)

  const steps: Step[] = [
    {
      id: 'forward',
      name: '前向传播',
      color: 'blue',
      totalTime: 35,
      substeps: [
        { name: 'Embedding Lookup', time: 5, memory: 200, description: '将 token IDs 转为 embedding vectors' },
        { name: 'Attention Layers', time: 20, memory: 1500, description: '多层 self-attention 计算' },
        { name: 'Feed-Forward', time: 8, memory: 800, description: 'FFN 层计算' },
        { name: 'Output Projection', time: 2, memory: 100, description: '最终输出层' }
      ]
    },
    {
      id: 'loss',
      name: '损失计算',
      color: 'purple',
      totalTime: 5,
      substeps: [
        { name: 'Cross-Entropy Loss', time: 3, memory: 50, description: '计算 logits 与 labels 的交叉熵' },
        { name: 'Loss Reduction', time: 2, memory: 10, description: '聚合批次损失（mean/sum）' }
      ]
    },
    {
      id: 'backward',
      name: '反向传播',
      color: 'pink',
      totalTime: 42,
      substeps: [
        { name: 'Output Gradient', time: 5, memory: 100, description: '计算输出层梯度' },
        { name: 'Backprop FFN', time: 10, memory: 800, description: 'FFN 层梯度回传' },
        { name: 'Backprop Attention', time: 25, memory: 1500, description: 'Attention 层梯度回传（最耗时）' },
        { name: 'Embedding Gradient', time: 2, memory: 200, description: 'Embedding 层梯度' }
      ]
    },
    {
      id: 'optimize',
      name: '参数更新',
      color: 'green',
      totalTime: 18,
      substeps: [
        { name: 'Gradient Clipping', time: 3, memory: 50, description: '梯度裁剪（防止梯度爆炸）' },
        { name: 'Adam Update', time: 12, memory: 600, description: 'Adam 优化器更新参数' },
        { name: 'Zero Gradients', time: 2, memory: 10, description: '清零梯度存储' },
        { name: 'LR Scheduler', time: 1, memory: 5, description: '更新学习率' }
      ]
    }
  ]

  const currentStep = steps[activeStep]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white">
          训练步骤时间分解
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          单个 training step 的子步骤详细分析
        </p>
      </div>

      {/* Step Navigation */}
      <div className="grid grid-cols-4 gap-2 mb-6">
        {steps.map((step, idx) => (
          <button
            key={step.id}
            onClick={() => setActiveStep(idx)}
            className={`p-3 rounded-lg font-semibold transition-all ${
              activeStep === idx
                ? `bg-${step.color}-500 text-white shadow-lg`
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
            }`}
          >
            <div className="text-sm">{step.name}</div>
            <div className="text-xs opacity-75 mt-1">{step.totalTime}ms</div>
          </button>
        ))}
      </div>

      {/* Substeps Detail */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="space-y-3"
        >
          {currentStep.substeps.map((substep, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
            >
              <div className="flex items-start justify-between gap-4 mb-2">
                <div className="flex-1">
                  <h4 className="font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full bg-${currentStep.color}-500`}></span>
                    {substep.name}
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                    {substep.description}
                  </p>
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4 mt-3">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-500" />
                  <div>
                    <div className="text-xs text-slate-500">时间</div>
                    <div className="font-bold text-slate-900 dark:text-white">{substep.time}ms</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <MemoryStick className="w-4 h-4 text-green-500" />
                  <div>
                    <div className="text-xs text-slate-500">显存</div>
                    <div className="font-bold text-slate-900 dark:text-white">{substep.memory}MB</div>
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mt-3">
                <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                  <span>时间占比</span>
                  <span>{((substep.time / currentStep.totalTime) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(substep.time / currentStep.totalTime) * 100}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                    className={`h-2 bg-${currentStep.color}-500 rounded-full`}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </AnimatePresence>

      {/* Total Statistics */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-blue-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">总时间</span>
          </div>
          <div className="text-2xl font-bold text-blue-500">100ms</div>
          <div className="text-xs text-slate-500 mt-1">35+5+42+18</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <MemoryStick className="w-5 h-5 text-green-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">峰值显存</span>
          </div>
          <div className="text-2xl font-bold text-green-500">2.6GB</div>
          <div className="text-xs text-slate-500 mt-1">Attention 层最高</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-purple-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">并行度</span>
          </div>
          <div className="text-2xl font-bold text-purple-500">85%</div>
          <div className="text-xs text-slate-500 mt-1">GPU 利用率</div>
        </div>
      </div>
    </div>
  )
}

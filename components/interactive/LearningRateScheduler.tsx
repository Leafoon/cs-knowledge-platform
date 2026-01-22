'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingDown, TrendingUp, Activity } from 'lucide-react'

interface SchedulerConfig {
  id: string
  name: string
  description: string
  color: string
  generatePoints: (steps: number, warmup: number, maxLR: number) => number[]
}

export default function LearningRateScheduler() {
  const [selectedScheduler, setSelectedScheduler] = useState('linear')
  const [warmupSteps, setWarmupSteps] = useState(500)
  const [maxLR, setMaxLR] = useState(5e-5)
  const totalSteps = 5000

  const schedulers: SchedulerConfig[] = [
    {
      id: 'linear',
      name: 'Linear Decay',
      description: 'Warmup 后线性衰减到 0',
      color: 'blue',
      generatePoints: (steps, warmup, lr) => {
        const points = []
        for (let i = 0; i <= steps; i += steps / 50) {
          if (i < warmup) {
            points.push((i / warmup) * lr)
          } else {
            points.push(lr * (1 - (i - warmup) / (steps - warmup)))
          }
        }
        return points
      }
    },
    {
      id: 'cosine',
      name: 'Cosine Annealing',
      description: 'Warmup 后余弦衰减',
      color: 'purple',
      generatePoints: (steps, warmup, lr) => {
        const points = []
        for (let i = 0; i <= steps; i += steps / 50) {
          if (i < warmup) {
            points.push((i / warmup) * lr)
          } else {
            const progress = (i - warmup) / (steps - warmup)
            points.push(lr * 0.5 * (1 + Math.cos(Math.PI * progress)))
          }
        }
        return points
      }
    },
    {
      id: 'constant_warmup',
      name: 'Constant with Warmup',
      description: 'Warmup 后保持恒定',
      color: 'green',
      generatePoints: (steps, warmup, lr) => {
        const points = []
        for (let i = 0; i <= steps; i += steps / 50) {
          if (i < warmup) {
            points.push((i / warmup) * lr)
          } else {
            points.push(lr)
          }
        }
        return points
      }
    },
    {
      id: 'polynomial',
      name: 'Polynomial Decay',
      description: 'Warmup 后多项式衰减',
      color: 'orange',
      generatePoints: (steps, warmup, lr) => {
        const points = []
        const power = 2
        for (let i = 0; i <= steps; i += steps / 50) {
          if (i < warmup) {
            points.push((i / warmup) * lr)
          } else {
            const progress = (i - warmup) / (steps - warmup)
            points.push(lr * Math.pow(1 - progress, power))
          }
        }
        return points
      }
    }
  ]

  const currentScheduler = schedulers.find(s => s.id === selectedScheduler)!
  const lrPoints = currentScheduler.generatePoints(totalSteps, warmupSteps, maxLR)

  const maxValue = Math.max(...lrPoints)
  const minValue = Math.min(...lrPoints)

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-teal-500" />
          学习率调度策略对比
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          不同 lr_scheduler_type 的学习率变化曲线
        </p>
      </div>

      {/* Scheduler Selection */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-6">
        {schedulers.map(scheduler => (
          <button
            key={scheduler.id}
            onClick={() => setSelectedScheduler(scheduler.id)}
            className={`p-3 rounded-lg font-semibold transition-all text-left ${
              selectedScheduler === scheduler.id
                ? `bg-${scheduler.color}-500 text-white shadow-lg`
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
            }`}
          >
            <div className="text-sm font-bold">{scheduler.name}</div>
            <div className="text-xs opacity-75 mt-1">{scheduler.description}</div>
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <label className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between mb-2">
            <span>Warmup Steps</span>
            <span className="text-blue-500">{warmupSteps}</span>
          </label>
          <input
            type="range"
            min="0"
            max="2000"
            step="100"
            value={warmupSteps}
            onChange={(e) => setWarmupSteps(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <label className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between mb-2">
            <span>Max Learning Rate</span>
            <span className="text-green-500">{maxLR.toExponential(1)}</span>
          </label>
          <input
            type="range"
            min="1e-6"
            max="1e-4"
            step="1e-6"
            value={maxLR}
            onChange={(e) => setMaxLR(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Learning Rate Curve */}
      <div className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 mb-6">
        <div className="relative h-64">
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 bottom-0 w-16 flex flex-col justify-between text-xs text-slate-500">
            <span>{maxValue.toExponential(1)}</span>
            <span>{(maxValue * 0.5).toExponential(1)}</span>
            <span>0</span>
          </div>

          {/* Chart area */}
          <div className="ml-16 h-full relative">
            {/* Grid lines */}
            <div className="absolute inset-0">
              {[0, 25, 50, 75, 100].map(percent => (
                <div
                  key={percent}
                  className="absolute w-full border-t border-slate-200 dark:border-slate-700"
                  style={{ top: `${percent}%` }}
                />
              ))}
            </div>

            {/* Warmup zone */}
            {warmupSteps > 0 && (
              <div
                className="absolute top-0 bottom-0 bg-blue-100 dark:bg-blue-900/20 border-r-2 border-dashed border-blue-300 dark:border-blue-700"
                style={{ width: `${(warmupSteps / totalSteps) * 100}%` }}
              >
                <span className="absolute top-2 left-2 text-xs font-semibold text-blue-600 dark:text-blue-400">
                  Warmup
                </span>
              </div>
            )}

            {/* Learning rate curve */}
            <svg className="absolute inset-0 w-full h-full">
              <motion.path
                d={`M ${lrPoints.map((lr, i) => {
                  const x = (i / (lrPoints.length - 1)) * 100
                  const y = 100 - ((lr - minValue) / (maxValue - minValue)) * 100
                  return `${x}% ${y}%`
                }).join(' L ')}`}
                fill="none"
                stroke={`var(--color-${currentScheduler.color}-500)`}
                strokeWidth="3"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1 }}
              />
            </svg>
          </div>

          {/* X-axis */}
          <div className="absolute bottom-0 left-16 right-0 flex justify-between text-xs text-slate-500 mt-2">
            <span>0</span>
            <span>{totalSteps / 2}</span>
            <span>{totalSteps} steps</span>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">最大值</span>
          </div>
          <div className="text-xl font-bold text-green-500">{maxValue.toExponential(2)}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-red-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">最终值</span>
          </div>
          <div className="text-xl font-bold text-red-500">{lrPoints[lrPoints.length - 1].toExponential(2)}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">Warmup</span>
          </div>
          <div className="text-xl font-bold text-blue-500">{warmupSteps}</div>
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">配置代码</div>
        <pre className="text-sm text-slate-100 font-mono overflow-auto">
          <code>{`training_args = TrainingArguments(
    lr_scheduler_type="${selectedScheduler}",
    learning_rate=${maxLR.toExponential(1)},
    warmup_steps=${warmupSteps},
    max_steps=${totalSteps}
)`}</code>
        </pre>
      </div>
    </div>
  )
}

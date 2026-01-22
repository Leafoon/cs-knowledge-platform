'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingDown, Award, Zap, Eye, EyeOff } from 'lucide-react'

interface MetricData {
  epoch: number
  trainLoss: number
  evalLoss: number
  accuracy: number
  learningRate: number
}

export default function TrainingMetricsPlot() {
  const [selectedMetric, setSelectedMetric] = useState<'loss' | 'accuracy' | 'lr'>('loss')
  const [showTrain, setShowTrain] = useState(true)
  const [showEval, setShowEval] = useState(true)

  // Simulated training data
  const data: MetricData[] = [
    { epoch: 0, trainLoss: 2.3, evalLoss: 2.28, accuracy: 0.52, learningRate: 5e-5 },
    { epoch: 1, trainLoss: 1.15, evalLoss: 1.22, accuracy: 0.68, learningRate: 4.5e-5 },
    { epoch: 2, trainLoss: 0.68, evalLoss: 0.75, accuracy: 0.78, learningRate: 4e-5 },
    { epoch: 3, trainLoss: 0.45, evalLoss: 0.51, accuracy: 0.85, learningRate: 3.5e-5 },
    { epoch: 4, trainLoss: 0.32, evalLoss: 0.42, accuracy: 0.89, learningRate: 3e-5 },
    { epoch: 5, trainLoss: 0.24, evalLoss: 0.38, accuracy: 0.91, learningRate: 2.5e-5 },
    { epoch: 6, trainLoss: 0.18, evalLoss: 0.36, accuracy: 0.92, learningRate: 2e-5 },
    { epoch: 7, trainLoss: 0.14, evalLoss: 0.35, accuracy: 0.93, learningRate: 1.5e-5 },
    { epoch: 8, trainLoss: 0.11, evalLoss: 0.34, accuracy: 0.935, learningRate: 1e-5 },
    { epoch: 9, trainLoss: 0.09, evalLoss: 0.34, accuracy: 0.936, learningRate: 5e-6 },
    { epoch: 10, trainLoss: 0.08, evalLoss: 0.35, accuracy: 0.934, learningRate: 1e-6 }
  ]

  const maxLoss = Math.max(...data.map(d => Math.max(d.trainLoss, d.evalLoss)))
  const minLoss = Math.min(...data.map(d => Math.min(d.trainLoss, d.evalLoss)))

  const getYValue = (d: MetricData) => {
    if (selectedMetric === 'loss') return d.trainLoss
    if (selectedMetric === 'accuracy') return d.accuracy * 100
    return d.learningRate * 1e6
  }

  const getMaxY = () => {
    if (selectedMetric === 'loss') return maxLoss
    if (selectedMetric === 'accuracy') return 100
    return 50
  }

  const bestEpoch = selectedMetric === 'loss' 
    ? data.reduce((min, d) => d.evalLoss < min.evalLoss ? d : min, data[0])
    : data.reduce((max, d) => d.accuracy > max.accuracy ? d : max, data[0])

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-emerald-500" />
            训练指标可视化
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            实时监控训练过程
          </p>
        </div>
      </div>

      {/* Metric Selector */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSelectedMetric('loss')}
          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-all ${
            selectedMetric === 'loss'
              ? 'bg-red-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          Loss
        </button>
        <button
          onClick={() => setSelectedMetric('accuracy')}
          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-all ${
            selectedMetric === 'accuracy'
              ? 'bg-green-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          Accuracy
        </button>
        <button
          onClick={() => setSelectedMetric('lr')}
          className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-all ${
            selectedMetric === 'lr'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          Learning Rate
        </button>
      </div>

      {/* Toggle Lines */}
      {selectedMetric === 'loss' && (
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setShowTrain(!showTrain)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              showTrain
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
            }`}
          >
            {showTrain ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            Train Loss
          </button>
          <button
            onClick={() => setShowEval(!showEval)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              showEval
                ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
            }`}
          >
            {showEval ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            Eval Loss
          </button>
        </div>
      )}

      {/* Chart */}
      <div className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 mb-6">
        <div className="relative h-64">
          {/* Y-axis */}
          <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-slate-500">
            <span>{getMaxY().toFixed(selectedMetric === 'lr' ? 0 : 2)}</span>
            <span>{(getMaxY() * 0.5).toFixed(selectedMetric === 'lr' ? 0 : 2)}</span>
            <span>0</span>
          </div>

          {/* Chart area */}
          <div className="ml-12 h-full relative">
            {/* Grid */}
            <div className="absolute inset-0">
              {[0, 25, 50, 75, 100].map(percent => (
                <div
                  key={percent}
                  className="absolute w-full border-t border-slate-200 dark:border-slate-700"
                  style={{ top: `${percent}%` }}
                />
              ))}
              {[0, 25, 50, 75, 100].map(percent => (
                <div
                  key={percent}
                  className="absolute h-full border-l border-slate-200 dark:border-slate-700"
                  style={{ left: `${percent}%` }}
                />
              ))}
            </div>

            {/* Data points and lines */}
            <svg className="absolute inset-0 w-full h-full">
              {/* Train Loss Line */}
              {selectedMetric === 'loss' && showTrain && (
                <motion.path
                  d={`M ${data.map((d, i) => {
                    const x = (i / (data.length - 1)) * 100
                    const y = 100 - (d.trainLoss / maxLoss) * 100
                    return `${x}% ${y}%`
                  }).join(' L ')}`}
                  fill="none"
                  stroke="rgb(59, 130, 246)"
                  strokeWidth="2"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1 }}
                />
              )}

              {/* Eval Loss Line */}
              {selectedMetric === 'loss' && showEval && (
                <motion.path
                  d={`M ${data.map((d, i) => {
                    const x = (i / (data.length - 1)) * 100
                    const y = 100 - (d.evalLoss / maxLoss) * 100
                    return `${x}% ${y}%`
                  }).join(' L ')}`}
                  fill="none"
                  stroke="rgb(249, 115, 22)"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1, delay: 0.2 }}
                />
              )}

              {/* Accuracy/LR Line */}
              {selectedMetric !== 'loss' && (
                <motion.path
                  d={`M ${data.map((d, i) => {
                    const x = (i / (data.length - 1)) * 100
                    const value = selectedMetric === 'accuracy' ? d.accuracy * 100 : d.learningRate * 1e6
                    const y = 100 - (value / getMaxY()) * 100
                    return `${x}% ${y}%`
                  }).join(' L ')}`}
                  fill="none"
                  stroke={selectedMetric === 'accuracy' ? 'rgb(34, 197, 94)' : 'rgb(59, 130, 246)'}
                  strokeWidth="2"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1 }}
                />
              )}

              {/* Data points */}
              {data.map((d, i) => {
                const x = (i / (data.length - 1)) * 100
                let y = 0
                if (selectedMetric === 'loss') {
                  y = 100 - (d.evalLoss / maxLoss) * 100
                } else if (selectedMetric === 'accuracy') {
                  y = 100 - (d.accuracy * 100 / getMaxY()) * 100
                } else {
                  y = 100 - (d.learningRate * 1e6 / getMaxY()) * 100
                }
                return (
                  <motion.circle
                    key={i}
                    cx={`${x}%`}
                    cy={`${y}%`}
                    r="4"
                    fill={i === bestEpoch.epoch ? 'rgb(34, 197, 94)' : 'rgb(148, 163, 184)'}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                  />
                )
              })}
            </svg>

            {/* Best epoch marker */}
            <div
              className="absolute top-0 bottom-0 border-l-2 border-dashed border-green-500"
              style={{ left: `${(bestEpoch.epoch / (data.length - 1)) * 100}%` }}
            >
              <span className="absolute -top-6 left-2 text-xs font-semibold text-green-600 dark:text-green-400 whitespace-nowrap">
                Best: Epoch {bestEpoch.epoch}
              </span>
            </div>
          </div>

          {/* X-axis */}
          <div className="absolute -bottom-6 left-12 right-0 flex justify-between text-xs text-slate-500">
            <span>0</span>
            <span>5</span>
            <span>10 epochs</span>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Award className="w-4 h-4 text-green-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">最佳 Epoch</span>
          </div>
          <div className="text-2xl font-bold text-green-500">{bestEpoch.epoch}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-red-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">最低 Loss</span>
          </div>
          <div className="text-2xl font-bold text-red-500">{bestEpoch.evalLoss.toFixed(3)}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-blue-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">最高准确率</span>
          </div>
          <div className="text-2xl font-bold text-blue-500">{(bestEpoch.accuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-purple-500" />
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">改善</span>
          </div>
          <div className="text-2xl font-bold text-purple-500">
            {((1 - data[data.length - 1].evalLoss / data[0].evalLoss) * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  )
}

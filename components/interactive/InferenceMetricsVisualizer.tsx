'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Gauge, Clock, Zap, TrendingUp } from 'lucide-react'

interface Metric {
  name: string
  value: number
  unit: string
  description: string
  icon: React.ReactNode
  color: string
}

export default function InferenceMetricsVisualizer() {
  const [selectedConfig, setSelectedConfig] = useState<'baseline' | 'optimized'>('baseline')

  const configs = {
    baseline: {
      name: '基线配置 (FP16)',
      metrics: [
        { name: 'TTFT', value: 450, unit: 'ms', description: 'Time to First Token - 首token延迟', icon: <Clock className="w-5 h-5" />, color: 'blue' },
        { name: 'TPS', value: 22, unit: 'tokens/s', description: 'Tokens Per Second - 吞吐量', icon: <Zap className="w-5 h-5" />, color: 'green' },
        { name: 'Latency (Avg)', value: 890, unit: 'ms', description: '平均端到端延迟', icon: <Gauge className="w-5 h-5" />, color: 'purple' },
        { name: 'Latency (P95)', value: 1250, unit: 'ms', description: '95分位延迟', icon: <TrendingUp className="w-5 h-5" />, color: 'orange' },
      ]
    },
    optimized: {
      name: '优化配置 (Flash Attention + Compile + INT8)',
      metrics: [
        { name: 'TTFT', value: 180, unit: 'ms', description: 'Time to First Token - 首token延迟', icon: <Clock className="w-5 h-5" />, color: 'blue' },
        { name: 'TPS', value: 68, unit: 'tokens/s', description: 'Tokens Per Second - 吞吐量', icon: <Zap className="w-5 h-5" />, color: 'green' },
        { name: 'Latency (Avg)', value: 320, unit: 'ms', description: '平均端到端延迟', icon: <Gauge className="w-5 h-5" />, color: 'purple' },
        { name: 'Latency (P95)', value: 480, unit: 'ms', description: '95分位延迟', icon: <TrendingUp className="w-5 h-5" />, color: 'orange' },
      ]
    }
  }

  const currentConfig = configs[selectedConfig]
  const baselineConfig = configs.baseline

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Gauge className="w-8 h-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-slate-800">推理性能指标对比</h3>
      </div>

      {/* 配置选择器 */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => setSelectedConfig('baseline')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            selectedConfig === 'baseline'
              ? 'border-blue-600 bg-blue-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-blue-300'
          }`}
        >
          <div className={`font-bold ${selectedConfig === 'baseline' ? 'text-blue-900' : 'text-slate-700'}`}>
            {configs.baseline.name}
          </div>
        </button>
        <button
          onClick={() => setSelectedConfig('optimized')}
          className={`flex-1 p-4 rounded-lg border-2 transition-all ${
            selectedConfig === 'optimized'
              ? 'border-green-600 bg-green-50 shadow-md'
              : 'border-slate-200 bg-white hover:border-green-300'
          }`}
        >
          <div className={`font-bold ${selectedConfig === 'optimized' ? 'text-green-900' : 'text-slate-700'}`}>
            {configs.optimized.name}
          </div>
        </button>
      </div>

      {/* 指标卡片 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {currentConfig.metrics.map((metric, idx) => {
          const baselineMetric = baselineConfig.metrics[idx]
          const improvement = ((baselineMetric.value - metric.value) / baselineMetric.value) * 100
          const isPositive = improvement > 0 || (metric.name === 'TPS' && metric.value > baselineMetric.value)

          return (
            <motion.div
              key={metric.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white p-5 rounded-lg shadow-lg"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`text-${metric.color}-600`}>
                    {metric.icon}
                  </div>
                  <div>
                    <div className="font-bold text-slate-800">{metric.name}</div>
                    <div className="text-xs text-slate-500">{metric.description}</div>
                  </div>
                </div>
              </div>

              {/* 数值显示 */}
              <div className="mb-3">
                <motion.div
                  initial={{ scale: 0.5 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: idx * 0.1 + 0.2, type: 'spring' }}
                  className={`text-4xl font-bold text-${metric.color}-600`}
                >
                  {metric.value}
                  <span className="text-lg text-slate-600 ml-2">{metric.unit}</span>
                </motion.div>
              </div>

              {/* 进度条 */}
              <div className="relative h-4 bg-slate-100 rounded-full overflow-hidden mb-3">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(metric.value / Math.max(baselineMetric.value, metric.value)) * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.1 + 0.3 }}
                  className={`h-full bg-gradient-to-r from-${metric.color}-400 to-${metric.color}-600`}
                />
              </div>

              {/* 提升百分比 */}
              {selectedConfig === 'optimized' && (
                <div className={`text-sm font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {metric.name === 'TPS' ? (
                    <>↑ {((metric.value / baselineMetric.value - 1) * 100).toFixed(0)}% 提升</>
                  ) : (
                    <>↓ {improvement.toFixed(0)}% 降低</>
                  )}
                </div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* 对比表格 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">详细对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300">
              <th className="text-left py-2 px-4">指标</th>
              <th className="text-right py-2 px-4">基线配置</th>
              <th className="text-right py-2 px-4">优化配置</th>
              <th className="text-right py-2 px-4">提升</th>
            </tr>
          </thead>
          <tbody>
            {baselineConfig.metrics.map((baseMetric, idx) => {
              const optMetric = configs.optimized.metrics[idx]
              const improvement = baseMetric.name === 'TPS'
                ? ((optMetric.value / baseMetric.value - 1) * 100)
                : ((baseMetric.value - optMetric.value) / baseMetric.value * 100)

              return (
                <tr key={idx} className="border-b border-slate-200">
                  <td className="py-3 px-4 font-medium">{baseMetric.name}</td>
                  <td className="py-3 px-4 text-right font-mono">{baseMetric.value} {baseMetric.unit}</td>
                  <td className="py-3 px-4 text-right font-mono text-green-600 font-bold">
                    {optMetric.value} {optMetric.unit}
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-green-600">
                    {improvement > 0 ? '+' : ''}{improvement.toFixed(0)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* 说明 */}
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-slate-700">
        <div className="font-bold text-blue-800 mb-2">指标说明</div>
        <ul className="space-y-1">
          <li><strong>TTFT</strong>: 从发送请求到收到第一个token的时间（越低越好）</li>
          <li><strong>TPS</strong>: 每秒生成的token数量（越高越好）</li>
          <li><strong>Latency (Avg)</strong>: 生成50个token的平均耗时</li>
          <li><strong>Latency (P95)</strong>: 95%的请求在此时间内完成</li>
        </ul>
      </div>
    </div>
  )
}

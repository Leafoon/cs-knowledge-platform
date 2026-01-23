'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Cpu, TrendingDown, Lock } from 'lucide-react'

interface Method {
  name: string
  trainableParams: number
  memory: number
  performance: number
  color: string
  description: string
}

export default function PEFTMethodComparison() {
  const [selectedMethod, setSelectedMethod] = useState<string>('full')

  const methods: Record<string, Method> = {
    full: {
      name: '全量微调',
      trainableParams: 100,
      memory: 100,
      performance: 100,
      color: 'slate',
      description: '更新所有参数，最高性能但显存占用大'
    },
    lora: {
      name: 'LoRA',
      trainableParams: 0.5,
      memory: 30,
      performance: 99,
      color: 'blue',
      description: '低秩矩阵分解，性价比最高'
    },
    qlora: {
      name: 'QLoRA',
      trainableParams: 0.5,
      memory: 12,
      performance: 98,
      color: 'purple',
      description: '4-bit量化 + LoRA，显存最优'
    },
    prefix: {
      name: 'Prefix Tuning',
      trainableParams: 0.1,
      memory: 25,
      performance: 95,
      color: 'green',
      description: '添加可学习前缀，适合生成任务'
    },
    ia3: {
      name: '(IA)³',
      trainableParams: 0.01,
      memory: 20,
      performance: 94,
      color: 'orange',
      description: '激活缩放，参数量最少'
    }
  }

  const selected = methods[selectedMethod]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-blue-500" />
          PEFT 方法对比
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          点击选择不同的参数高效微调方法
        </p>
      </div>

      {/* Method Selector */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        {Object.entries(methods).map(([key, method]) => (
          <button
            key={key}
            onClick={() => setSelectedMethod(key)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedMethod === key
                ? `border-${method.color}-500 bg-${method.color}-50 dark:bg-${method.color}-900/20`
                : 'border-slate-200 dark:border-slate-700 hover:border-slate-300'
            }`}
          >
            <div className={`text-sm font-bold text-${method.color}-600 dark:text-${method.color}-400`}>
              {method.name}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {method.trainableParams}% 参数
            </div>
          </button>
        ))}
      </div>

      {/* Metrics */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {/* Trainable Parameters */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <Lock className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              可训练参数
            </span>
          </div>
          <div className="mb-2">
            <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
              <span>0%</span>
              <span>{selected.trainableParams}%</span>
              <span>100%</span>
            </div>
            <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${selected.trainableParams}%` }}
                transition={{ duration: 0.5 }}
                className={`h-full bg-${selected.color}-500`}
              />
            </div>
          </div>
          <div className={`text-2xl font-bold text-${selected.color}-600 dark:text-${selected.color}-400`}>
            {selected.trainableParams}%
          </div>
        </div>

        {/* Memory Usage */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-4 h-4 text-purple-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              显存占用
            </span>
          </div>
          <div className="mb-2">
            <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
              <span>0%</span>
              <span>{selected.memory}%</span>
              <span>100%</span>
            </div>
            <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${selected.memory}%` }}
                transition={{ duration: 0.5 }}
                className="h-full bg-purple-500"
              />
            </div>
          </div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {selected.memory}%
          </div>
        </div>

        {/* Performance */}
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-green-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              相对性能
            </span>
          </div>
          <div className="mb-2">
            <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
              <span>90%</span>
              <span>{selected.performance}%</span>
              <span>100%</span>
            </div>
            <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(selected.performance - 90) * 10}%` }}
                transition={{ duration: 0.5 }}
                className="h-full bg-green-500"
              />
            </div>
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {selected.performance}%
          </div>
        </div>
      </div>

      {/* Description */}
      <div className={`p-4 bg-${selected.color}-50 dark:bg-${selected.color}-900/20 border border-${selected.color}-200 dark:border-${selected.color}-800 rounded-lg`}>
        <div className={`text-sm font-semibold text-${selected.color}-700 dark:text-${selected.color}-300 mb-1`}>
          {selected.name}
        </div>
        <div className={`text-sm text-${selected.color}-600 dark:text-${selected.color}-400`}>
          {selected.description}
        </div>
      </div>

      {/* Summary Table */}
      <div className="mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-100 dark:bg-slate-800">
            <tr>
              <th className="px-4 py-2 text-left">方法</th>
              <th className="px-4 py-2 text-right">可训练参数</th>
              <th className="px-4 py-2 text-right">显存占用</th>
              <th className="px-4 py-2 text-right">相对性能</th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-slate-900">
            {Object.values(methods).map((method) => (
              <tr
                key={method.name}
                className={`border-b border-slate-200 dark:border-slate-700 ${
                  method.name === selected.name ? `bg-${method.color}-50 dark:bg-${method.color}-900/10` : ''
                }`}
              >
                <td className="px-4 py-2 font-medium">{method.name}</td>
                <td className="px-4 py-2 text-right">{method.trainableParams}%</td>
                <td className="px-4 py-2 text-right">{method.memory}%</td>
                <td className="px-4 py-2 text-right">{method.performance}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

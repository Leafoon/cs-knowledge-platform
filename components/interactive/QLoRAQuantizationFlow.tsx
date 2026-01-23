'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, Layers, ArrowRight } from 'lucide-react'

interface Step {
  id: number
  title: string
  description: string
  color: string
}

export default function QLoRAQuantizationFlow() {
  const [currentStep, setCurrentStep] = useState(0)

  const steps: Step[] = [
    {
      id: 0,
      title: '1. 加载 FP16 权重',
      description: '原始模型权重为 FP16 精度（7B 模型 = 14 GB）',
      color: 'blue'
    },
    {
      id: 1,
      title: '2. NF4 量化',
      description: '使用 NormalFloat4 数据类型量化到 4-bit（7B = 3.5 GB）',
      color: 'purple'
    },
    {
      id: 2,
      title: '3. 双重量化',
      description: '量化常数本身也量化为 8-bit，进一步节省显存',
      color: 'violet'
    },
    {
      id: 3,
      title: '4. 添加 LoRA 适配器',
      description: 'LoRA 权重保持 FP16 精度，仅占 ~0.5 GB',
      color: 'green'
    },
    {
      id: 4,
      title: '5. 混合精度训练',
      description: '前向传播时反量化为 BF16，backward 仅更新 LoRA',
      color: 'orange'
    }
  ]

  const memoryData = [
    { step: '原始模型', size: 14, color: 'bg-blue-500' },
    { step: 'NF4量化', size: 3.5, color: 'bg-purple-500' },
    { step: '双重量化', size: 3.3, color: 'bg-violet-500' },
    { step: '+ LoRA', size: 3.8, color: 'bg-green-500' },
    { step: '训练时', size: 6.5, color: 'bg-orange-500' }
  ]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-purple-500" />
          QLoRA 量化流程
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          4-bit 量化 + LoRA 训练的完整流程
        </p>
      </div>

      {/* Step Progress */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          {steps.map((step, idx) => (
            <React.Fragment key={step.id}>
              <button
                onClick={() => setCurrentStep(idx)}
                className={`relative flex flex-col items-center p-3 rounded-lg transition-all ${
                  currentStep === idx
                    ? `bg-${step.color}-100 dark:bg-${step.color}-900/30 scale-110`
                    : currentStep > idx
                    ? 'bg-slate-100 dark:bg-slate-800'
                    : 'bg-slate-50 dark:bg-slate-900'
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                    currentStep === idx
                      ? `bg-${step.color}-500 text-white`
                      : currentStep > idx
                      ? 'bg-green-500 text-white'
                      : 'bg-slate-300 dark:bg-slate-700 text-slate-600'
                  }`}
                >
                  {currentStep > idx ? '✓' : idx + 1}
                </div>
                <span className="text-xs text-center max-w-20 text-slate-700 dark:text-slate-300">
                  {step.title.split('.')[1]?.trim()}
                </span>
              </button>
              {idx < steps.length - 1 && (
                <ArrowRight className={`w-6 h-6 ${
                  currentStep > idx ? 'text-green-500' : 'text-slate-300 dark:text-slate-600'
                }`} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Step Details */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="mb-6"
        >
          <div className={`p-6 bg-${steps[currentStep].color}-50 dark:bg-${steps[currentStep].color}-900/20 border border-${steps[currentStep].color}-200 dark:border-${steps[currentStep].color}-800 rounded-lg`}>
            <h4 className={`text-lg font-bold text-${steps[currentStep].color}-700 dark:text-${steps[currentStep].color}-300 mb-2`}>
              {steps[currentStep].title}
            </h4>
            <p className={`text-sm text-${steps[currentStep].color}-600 dark:text-${steps[currentStep].color}-400`}>
              {steps[currentStep].description}
            </p>

            {/* Step-specific content */}
            {currentStep === 1 && (
              <div className="mt-4 p-3 bg-white dark:bg-slate-800 rounded">
                <div className="text-xs font-mono text-slate-700 dark:text-slate-300">
                  NF4 量化级别 (16个): [-1.0, -0.696, ..., 0.0, ..., 0.723, 1.0]
                </div>
              </div>
            )}

            {currentStep === 3 && (
              <div className="mt-4 p-3 bg-white dark:bg-slate-800 rounded">
                <div className="text-xs text-slate-700 dark:text-slate-300">
                  <div>LoRA 配置: r=16, alpha=32</div>
                  <div>target_modules: [&quot;q_proj&quot;, &quot;k_proj&quot;, &quot;v_proj&quot;, &quot;o_proj&quot;]</div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Memory Comparison */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          显存占用变化（7B 模型）
        </h4>
        <div className="space-y-3">
          {memoryData.map((data, idx) => (
            <motion.div
              key={data.step}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="relative"
            >
              <div className="flex items-center gap-3">
                <div className="w-24 text-sm text-slate-700 dark:text-slate-300">
                  {data.step}
                </div>
                <div className="flex-1">
                  <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(data.size / 14) * 100}%` }}
                      transition={{ duration: 0.5, delay: idx * 0.1 }}
                      className={`h-full ${data.color} flex items-center justify-end pr-3`}
                    >
                      <span className="text-xs font-bold text-white">
                        {data.size} GB
                      </span>
                    </motion.div>
                  </div>
                </div>
                <div className="w-16 text-right text-sm font-bold text-slate-700 dark:text-slate-300">
                  {((data.size / 14) * 100).toFixed(0)}%
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">显存节省</div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            77%
          </div>
          <div className="text-xs text-slate-500 mt-1">14 GB → 3.2 GB</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">可训练参数</div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            0.06%
          </div>
          <div className="text-xs text-slate-500 mt-1">4.2M / 7B</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">性能损失</div>
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            &lt;2%
          </div>
          <div className="text-xs text-slate-500 mt-1">相比全量微调</div>
        </div>
      </div>

      {/* Navigation */}
      <div className="mt-6 flex justify-between">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 rounded-lg disabled:opacity-50"
        >
          ← 上一步
        </button>
        <button
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg disabled:opacity-50"
        >
          下一步 →
        </button>
      </div>
    </div>
  )
}

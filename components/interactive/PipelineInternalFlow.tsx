'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Code2, Cpu, FileOutput } from 'lucide-react'

interface Step {
  id: string
  title: string
  subtitle: string
  icon: React.ReactNode
  color: string
  input: string
  output: string
  code: string
  details: string[]
}

export default function PipelineInternalFlow() {
  const [activeStep, setActiveStep] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)

  const steps: Step[] = [
    {
      id: 'tokenization',
      title: 'Step 1: Tokenization',
      subtitle: '文本 → Token IDs',
      icon: <Code2 className="w-6 h-6" />,
      color: 'blue',
      input: '"I love transformers!"',
      output: '[101, 1045, 2293, 19081, 999, 102]',
      code: 'inputs = tokenizer(text, return_tensors="pt")',
      details: [
        '分词：将文本切分为 subword tokens',
        '转换：查找每个 token 的 ID',
        '添加特殊 token：[CLS] 和 [SEP]',
        '生成 attention_mask'
      ]
    },
    {
      id: 'model',
      title: 'Step 2: Model Forward',
      subtitle: 'Token IDs → Logits',
      icon: <Cpu className="w-6 h-6" />,
      color: 'purple',
      input: 'tensor([[101, 1045, ...]])',
      output: 'tensor([[-0.32, 0.89]])',
      code: 'outputs = model(**inputs)',
      details: [
        'Embedding：将 ID 转为向量',
        'Transformer 层：12 层 self-attention',
        'Pooling：聚合序列信息',
        '输出 logits：未归一化的分数'
      ]
    },
    {
      id: 'postprocess',
      title: 'Step 3: Post-processing',
      subtitle: 'Logits → 可读结果',
      icon: <FileOutput className="w-6 h-6" />,
      color: 'green',
      input: 'tensor([[-0.32, 0.89]])',
      output: '{"label": "POSITIVE", "score": 0.71}',
      code: 'result = pipeline.postprocess(outputs)',
      details: [
        'Softmax：转为概率分布',
        'Argmax：选择最高分类别',
        '标签映射：ID → 可读标签',
        '格式化：生成最终 JSON'
      ]
    }
  ]

  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setIsAnimating(true)
      setTimeout(() => {
        setActiveStep(activeStep + 1)
        setIsAnimating(false)
      }, 500)
    }
  }

  const handlePrev = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1)
    }
  }

  const currentStep = steps[activeStep]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-slate-900 dark:to-sky-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white">
          Pipeline 内部流程详解
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          3 步自动处理：Tokenization → Model → Post-processing
        </p>
      </div>

      {/* Step Progress */}
      <div className="flex items-center justify-between mb-8">
        {steps.map((step, idx) => (
          <React.Fragment key={step.id}>
            <button
              onClick={() => setActiveStep(idx)}
              className={`flex flex-col items-center gap-2 p-4 rounded-lg transition-all ${
                activeStep === idx
                  ? `bg-${step.color}-500 text-white shadow-lg scale-105`
                  : activeStep > idx
                  ? `bg-${step.color}-100 dark:bg-${step.color}-900/30 text-${step.color}-600 dark:text-${step.color}-400`
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
              }`}
            >
              <div className={`p-3 rounded-full ${
                activeStep === idx ? 'bg-white/20' : ''
              }`}>
                {step.icon}
              </div>
              <div className="text-xs font-semibold text-center">
                {step.title.split(':')[1]?.trim() || step.title}
              </div>
              {activeStep === idx && (
                <motion.div
                  layoutId="activeIndicator"
                  className="h-1 w-12 bg-white rounded-full"
                />
              )}
            </button>
            {idx < steps.length - 1 && (
              <ArrowRight className={`w-6 h-6 ${
                activeStep > idx ? `text-${step.color}-500` : 'text-slate-300 dark:text-slate-600'
              }`} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          {/* Title */}
          <div className="text-center">
            <h4 className={`text-2xl font-bold text-${currentStep.color}-600 dark:text-${currentStep.color}-400 mb-2`}>
              {currentStep.title}
            </h4>
            <p className="text-slate-600 dark:text-slate-400">{currentStep.subtitle}</p>
          </div>

          {/* Input/Output */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">输入</div>
              <code className="text-sm text-slate-900 dark:text-white font-mono break-all">
                {currentStep.input}
              </code>
            </div>
            <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">输出</div>
              <code className="text-sm text-slate-900 dark:text-white font-mono break-all">
                {currentStep.output}
              </code>
            </div>
          </div>

          {/* Code */}
          <div className="p-4 bg-slate-900 rounded-lg">
            <div className="text-xs text-slate-400 mb-2">代码</div>
            <code className="text-sm text-green-400 font-mono">{currentStep.code}</code>
          </div>

          {/* Details */}
          <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
            <div className="text-sm font-semibold text-slate-900 dark:text-white mb-3">详细步骤：</div>
            <div className="space-y-2">
              {currentStep.details.map((detail, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2"
                >
                  <div className={`w-6 h-6 rounded-full bg-${currentStep.color}-100 dark:bg-${currentStep.color}-900/30 text-${currentStep.color}-600 dark:text-${currentStep.color}-400 flex items-center justify-center text-xs font-bold shrink-0`}>
                    {idx + 1}
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400 pt-1">{detail}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button
          onClick={handlePrev}
          disabled={activeStep === 0}
          className="px-4 py-2 bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← 上一步
        </button>
        <button
          onClick={handleNext}
          disabled={activeStep === steps.length - 1}
          className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步 →
        </button>
      </div>
    </div>
  )
}

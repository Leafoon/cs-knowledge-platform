'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Database, Filter, Shuffle, ArrowRight, Code2, Cpu } from 'lucide-react'

interface Step {
  id: string
  title: string
  icon: React.ReactNode
  description: string
  code: string
  output: string
}

export default function DatasetPipeline() {
  const [activeStep, setActiveStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const steps: Step[] = [
    {
      id: 'load',
      title: 'load_dataset',
      icon: <Database className="w-6 h-6" />,
      description: '从 Hub 或本地加载数据集（内存映射）',
      code: `dataset = load_dataset("imdb", split="train")
print(f"Rows: {len(dataset)}")`,
      output: 'Rows: 25000\nMemory: ~50MB (索引+元数据)'
    },
    {
      id: 'map',
      title: 'map()',
      icon: <Code2 className="w-6 h-6" />,
      description: '批量应用 tokenization（多进程加速）',
      code: `tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    num_proc=4
)`,
      output: '进度: 100% | 25000/25000 [00:08<00:00, 3125 it/s]'
    },
    {
      id: 'filter',
      title: 'filter()',
      icon: <Filter className="w-6 h-6" />,
      description: '条件筛选（移除空文本、异常数据）',
      code: `filtered = tokenized.filter(
    lambda x: len(x["input_ids"]) > 10
)`,
      output: 'Filtered: 25000 → 24856 (-144 样本)'
    },
    {
      id: 'shuffle',
      title: 'shuffle()',
      icon: <Shuffle className="w-6 h-6" />,
      description: '随机打乱顺序（提升训练效果）',
      code: `shuffled = filtered.shuffle(seed=42)`,
      output: '✓ Shuffled with seed 42'
    },
    {
      id: 'format',
      title: 'set_format()',
      icon: <Cpu className="w-6 h-6" />,
      description: '转换为 PyTorch 张量格式',
      code: `shuffled.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)`,
      output: 'Format: torch.Tensor\nDevice: CPU'
    }
  ]

  const handlePlay = () => {
    if (isPlaying) return
    setIsPlaying(true)
    setActiveStep(0)

    const interval = setInterval(() => {
      setActiveStep(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(interval)
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, 2000)
  }

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-500" />
            数据集处理流程
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            从加载到训练准备的完整 Pipeline
          </p>
        </div>
        <button
          onClick={handlePlay}
          disabled={isPlaying}
          className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          {isPlaying ? '播放中...' : '▶ 播放流程'}
        </button>
      </div>

      {/* Pipeline Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 mb-6">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            {/* Step Card */}
            <motion.div
              initial={{ scale: 1 }}
              animate={{
                scale: activeStep === index ? 1.05 : 1,
                opacity: activeStep >= index ? 1 : 0.5
              }}
              className={`p-4 rounded-lg border-2 transition-all ${
                activeStep === index
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={`p-2 rounded-lg ${
                  activeStep === index ? 'bg-blue-500 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                }`}>
                  {step.icon}
                </div>
              </div>
              <h4 className="font-mono text-sm font-bold text-slate-900 dark:text-white mb-1">
                {step.title}
              </h4>
              <p className="text-xs text-slate-600 dark:text-slate-400 line-clamp-2">
                {step.description}
              </p>
              
              {/* Progress Indicator */}
              {activeStep === index && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 2 }}
                  className="h-1 bg-blue-500 rounded-full mt-3"
                />
              )}
            </motion.div>

            {/* Arrow */}
            {index < steps.length - 1 && (
              <div className="hidden lg:flex items-center justify-center">
                <motion.div
                  animate={{
                    opacity: activeStep > index ? 1 : 0.3,
                    x: activeStep === index ? [0, 5, 0] : 0
                  }}
                  transition={{ duration: 0.8, repeat: activeStep === index ? Infinity : 0 }}
                >
                  <ArrowRight className="w-6 h-6 text-blue-500" />
                </motion.div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Code & Output Display */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="grid md:grid-cols-2 gap-4"
        >
          {/* Code Block */}
          <div className="bg-slate-900 rounded-lg p-4 overflow-auto">
            <div className="text-xs text-slate-400 mb-2 font-semibold">代码示例</div>
            <pre className="text-sm text-slate-100 font-mono">
              <code>{steps[activeStep].code}</code>
            </pre>
          </div>

          {/* Output Block */}
          <div className="bg-slate-800 rounded-lg p-4 overflow-auto">
            <div className="text-xs text-green-400 mb-2 font-semibold">输出结果</div>
            <pre className="text-sm text-green-300 font-mono whitespace-pre-wrap">
              {steps[activeStep].output}
            </pre>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Step Navigation */}
      <div className="flex justify-center gap-2 mt-6">
        {steps.map((_, index) => (
          <button
            key={index}
            onClick={() => setActiveStep(index)}
            className={`w-3 h-3 rounded-full transition-all ${
              activeStep === index
                ? 'bg-blue-500 w-8'
                : 'bg-slate-300 dark:bg-slate-600'
            }`}
            aria-label={`Go to step ${index + 1}`}
          />
        ))}
      </div>

      {/* Statistics */}
      <div className="mt-6 grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-blue-500">5</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">处理步骤</div>
        </div>
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-green-500">~50MB</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">内存占用</div>
        </div>
        <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-purple-500">3125/s</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">处理速度</div>
        </div>
      </div>
    </div>
  )
}

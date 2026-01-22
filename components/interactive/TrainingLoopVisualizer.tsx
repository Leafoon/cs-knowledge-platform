'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw } from 'lucide-react'

interface TrainingStep {
  id: string
  name: string
  description: string
  color: string
  duration: number
}

export default function TrainingLoopVisualizer() {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [epoch, setEpoch] = useState(1)
  const [step, setStep] = useState(1)
  const [loss, setLoss] = useState(0.5234)

  const steps: TrainingStep[] = [
    { id: '1', name: '前向传播', description: 'outputs = model(**batch)', color: 'bg-blue-500', duration: 1000 },
    { id: '2', name: '计算损失', description: 'loss = outputs.loss', color: 'bg-purple-500', duration: 500 },
    { id: '3', name: '反向传播', description: 'loss.backward()', color: 'bg-pink-500', duration: 1200 },
    { id: '4', name: '梯度裁剪', description: 'clip_grad_norm_()', color: 'bg-orange-500', duration: 300 },
    { id: '5', name: '优化器更新', description: 'optimizer.step()', color: 'bg-green-500', duration: 600 },
    { id: '6', name: '清零梯度', description: 'optimizer.zero_grad()', color: 'bg-cyan-500', duration: 400 },
    { id: '7', name: '学习率调度', description: 'scheduler.step()', color: 'bg-indigo-500', duration: 200 }
  ]

  useEffect(() => {
    if (!isPlaying) return

    const timer = setTimeout(() => {
      if (currentStep < steps.length - 1) {
        setCurrentStep(currentStep + 1)
        setLoss(prev => Math.max(0.01, prev - 0.005 * Math.random()))
      } else {
        setCurrentStep(0)
        setStep(prev => prev + 1)
        if (step % 100 === 0) {
          setEpoch(prev => prev + 1)
        }
      }
    }, steps[currentStep].duration)

    return () => clearTimeout(timer)
  }, [isPlaying, currentStep, step, steps])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentStep(0)
    setEpoch(1)
    setStep(1)
    setLoss(0.5234)
  }

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-slate-900 dark:to-violet-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white">
            训练循环流程可视化
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Trainer.train() 内部执行的每一步
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handlePlayPause}
            className="p-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <button
            onClick={handleReset}
            className="p-3 bg-slate-500 hover:bg-slate-600 text-white rounded-lg transition-colors"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Training Info */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">Epoch</div>
          <div className="text-2xl font-bold text-blue-500">{epoch}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">Step</div>
          <div className="text-2xl font-bold text-green-500">{step}</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">Loss</div>
          <div className="text-2xl font-bold text-red-500">{loss.toFixed(4)}</div>
        </div>
      </div>

      {/* Steps Visualization */}
      <div className="space-y-3">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0.3, scale: 0.95 }}
            animate={{
              opacity: currentStep === index ? 1 : 0.5,
              scale: currentStep === index ? 1 : 0.95,
              borderColor: currentStep === index ? '#3b82f6' : '#e2e8f0'
            }}
            className="p-4 bg-white dark:bg-slate-800 rounded-lg border-2 transition-all"
          >
            <div className="flex items-center gap-4">
              {/* Step Number */}
              <div className={`w-10 h-10 rounded-full ${step.color} flex items-center justify-center text-white font-bold shrink-0`}>
                {index + 1}
              </div>

              {/* Step Info */}
              <div className="flex-1">
                <div className="font-semibold text-slate-900 dark:text-white">
                  {step.name}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400 font-mono">
                  {step.description}
                </div>
              </div>

              {/* Progress Bar */}
              {currentStep === index && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: 100 }}
                  transition={{ duration: step.duration / 1000 }}
                  className="h-2 bg-blue-500 rounded-full"
                  style={{ width: '100px' }}
                />
              )}

              {/* Checkmark */}
              {currentStep > index && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center text-white text-sm"
                >
                  ✓
                </motion.div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Code Preview */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">等效代码</div>
        <pre className="text-sm text-slate-100 font-mono overflow-auto">
          <code>{`for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(**batch)  # ${currentStep >= 0 ? '✓' : '...'}
        loss = outputs.loss       # ${currentStep >= 1 ? '✓' : '...'}
        loss.backward()           # ${currentStep >= 2 ? '✓' : '...'}
        clip_grad_norm_()         # ${currentStep >= 3 ? '✓' : '...'}
        optimizer.step()          # ${currentStep >= 4 ? '✓' : '...'}
        optimizer.zero_grad()     # ${currentStep >= 5 ? '✓' : '...'}
        scheduler.step()          # ${currentStep >= 6 ? '✓' : '...'}`}</code>
        </pre>
      </div>
    </div>
  )
}

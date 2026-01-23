'use client'

import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Zap, CheckCircle2, Circle, Code } from 'lucide-react'

type HookType = 
  | 'on_train_begin'
  | 'on_epoch_begin'
  | 'on_step_begin'
  | 'on_step_end'
  | 'on_log'
  | 'on_evaluate'
  | 'on_save'
  | 'on_epoch_end'
  | 'on_train_end'

interface Hook {
  name: HookType
  displayName: string
  description: string
  color: string
  triggered: boolean
}

const hooks: Hook[] = [
  {
    name: 'on_train_begin',
    displayName: 'on_train_begin',
    description: 'Initialize training state, log parameters',
    color: 'from-blue-500 to-cyan-500',
    triggered: false
  },
  {
    name: 'on_epoch_begin',
    displayName: 'on_epoch_begin',
    description: 'Epoch start, reset metrics',
    color: 'from-green-500 to-emerald-500',
    triggered: false
  },
  {
    name: 'on_step_begin',
    displayName: 'on_step_begin',
    description: 'Before forward pass',
    color: 'from-purple-500 to-pink-500',
    triggered: false
  },
  {
    name: 'on_step_end',
    displayName: 'on_step_end',
    description: 'After backward pass, monitor gradients',
    color: 'from-orange-500 to-red-500',
    triggered: false
  },
  {
    name: 'on_log',
    displayName: 'on_log',
    description: 'Log metrics to TensorBoard/W&B',
    color: 'from-yellow-500 to-amber-500',
    triggered: false
  },
  {
    name: 'on_evaluate',
    displayName: 'on_evaluate',
    description: 'Run evaluation loop, compute metrics',
    color: 'from-indigo-500 to-purple-500',
    triggered: false
  },
  {
    name: 'on_save',
    displayName: 'on_save',
    description: 'Save checkpoint to disk',
    color: 'from-teal-500 to-cyan-500',
    triggered: false
  },
  {
    name: 'on_epoch_end',
    displayName: 'on_epoch_end',
    description: 'Epoch complete, report summary',
    color: 'from-rose-500 to-pink-500',
    triggered: false
  },
  {
    name: 'on_train_end',
    displayName: 'on_train_end',
    description: 'Training complete, final cleanup',
    color: 'from-violet-500 to-purple-500',
    triggered: false
  }
]

const trainingSteps = [
  { hook: 'on_train_begin', step: 0, label: 'Start Training' },
  { hook: 'on_epoch_begin', step: 1, label: 'Epoch 1 Start' },
  { hook: 'on_step_begin', step: 2, label: 'Step 1 Start' },
  { hook: 'on_step_end', step: 3, label: 'Step 1 End' },
  { hook: 'on_log', step: 4, label: 'Log Metrics' },
  { hook: 'on_step_begin', step: 5, label: 'Step 2 Start' },
  { hook: 'on_step_end', step: 6, label: 'Step 2 End' },
  { hook: 'on_evaluate', step: 7, label: 'Evaluate' },
  { hook: 'on_save', step: 8, label: 'Save Checkpoint' },
  { hook: 'on_epoch_end', step: 9, label: 'Epoch 1 End' },
  { hook: 'on_epoch_begin', step: 10, label: 'Epoch 2 Start' },
  { hook: 'on_step_begin', step: 11, label: 'Step 3 Start' },
  { hook: 'on_step_end', step: 12, label: 'Step 3 End' },
  { hook: 'on_epoch_end', step: 13, label: 'Epoch 2 End' },
  { hook: 'on_train_end', step: 14, label: 'Training Complete' }
] as const

export default function TrainerHookFlow() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [triggeredHooks, setTriggeredHooks] = useState<Set<HookType>>(new Set())
  const [stepHistory, setStepHistory] = useState<string[]>([])

  // 播放/暂停
  const togglePlay = () => {
    if (currentStep >= trainingSteps.length - 1 && isPlaying) {
      reset()
      return
    }
    setIsPlaying(!isPlaying)
  }

  // 重置
  const reset = () => {
    setIsPlaying(false)
    setCurrentStep(0)
    setTriggeredHooks(new Set())
    setStepHistory([])
  }

  // 下一步
  const nextStep = useCallback(() => {
    if (currentStep < trainingSteps.length - 1) {
      const newStep = currentStep + 1
      const currentHook = trainingSteps[newStep].hook as HookType
      setCurrentStep(newStep)
      setTriggeredHooks(prev => new Set([...prev, currentHook]))
      setStepHistory(prev => [...prev, trainingSteps[newStep].label])
    } else {
      setIsPlaying(false)
    }
  }, [currentStep, trainingSteps])

  // 自动播放
  React.useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(nextStep, 800)
      return () => clearTimeout(timer)
    }
  }, [isPlaying, currentStep, nextStep])

  // 生成示例代码
  const generateCode = () => {
    const selectedHooks = Array.from(triggeredHooks)
    if (selectedHooks.length === 0) return 'Start the animation to see hooks in action!'

    let code = `from transformers import TrainerCallback, TrainerState, TrainerControl\n\n`
    code += `class MyCallback(TrainerCallback):\n`
    
    selectedHooks.slice(0, 3).forEach(hookName => {
      const hook = hooks.find(h => h.name === hookName)
      if (hook) {
        code += `    def ${hookName}(self, args, state, control, **kwargs):\n`
        code += `        """\n`
        code += `        ${hook.description}\n`
        code += `        """\n`
        
        // 添加示例逻辑
        if (hookName === 'on_train_begin') {
          code += `        print("🚀 Training started!")\n`
          code += `        print(f"Total epochs: {args.num_train_epochs}")\n`
        } else if (hookName === 'on_step_end') {
          code += `        # Monitor gradients\n`
          code += `        model = kwargs["model"]\n`
          code += `        total_norm = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5\n`
          code += `        print(f"Step {state.global_step}: grad_norm={total_norm:.4f}")\n`
        } else if (hookName === 'on_log') {
          code += `        # Custom logging\n`
          code += `        logs = kwargs.get("logs", {})\n`
          code += `        if "loss" in logs:\n`
          code += `            print(f"📊 Loss: {logs['loss']:.4f}")\n`
        } else if (hookName === 'on_evaluate') {
          code += `        # Early stopping check\n`
          code += `        metrics = kwargs.get("metrics", {})\n`
          code += `        eval_loss = metrics.get("eval_loss")\n`
          code += `        print(f"✅ Evaluation: loss={eval_loss:.4f}")\n`
        }
        
        code += `\n`
      }
    })

    code += `# Usage\n`
    code += `trainer = Trainer(\n`
    code += `    model=model,\n`
    code += `    args=training_args,\n`
    code += `    train_dataset=train_dataset,\n`
    code += `    callbacks=[MyCallback()]\n`
    code += `)\n`

    return code
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border-2 border-indigo-200">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-gray-800">Trainer Hook Flow Visualizer</h3>
            <p className="text-sm text-gray-600">Explore TrainerCallback execution order</p>
          </div>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={togglePlay}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2 transition-all"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause' : currentStep === 0 ? 'Start' : 'Resume'}
          </button>
          <button
            onClick={reset}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2 transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Hook Timeline */}
        <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
          <h4 className="font-bold text-gray-700 mb-4">Hook Execution Timeline</h4>
          
          <div className="space-y-2 max-h-[500px] overflow-y-auto">
            {trainingSteps.map((step, idx) => {
              const hook = hooks.find(h => h.name === step.hook)
              const isActive = idx === currentStep
              const isPast = idx < currentStep
              
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`relative flex items-center gap-3 p-3 rounded-lg border-2 transition-all ${
                    isActive
                      ? 'bg-gradient-to-r ' + hook!.color + ' text-white border-transparent scale-105 shadow-lg'
                      : isPast
                      ? 'bg-gray-100 border-gray-300 text-gray-700'
                      : 'bg-white border-gray-200 text-gray-500'
                  }`}
                >
                  {/* Step indicator */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                    isActive ? 'bg-white text-blue-600' : isPast ? 'bg-gray-300 text-white' : 'bg-gray-200 text-gray-400'
                  }`}>
                    {isPast ? <CheckCircle2 className="w-5 h-5" /> : idx + 1}
                  </div>
                  
                  {/* Hook info */}
                  <div className="flex-1">
                    <div className="font-semibold text-sm">{step.label}</div>
                    <div className={`text-xs ${isActive ? 'text-white/90' : 'text-gray-500'} font-mono`}>
                      {step.hook}()
                    </div>
                  </div>

                  {/* Animation pulse for active */}
                  {isActive && (
                    <motion.div
                      className="absolute right-3"
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                    >
                      <Circle className="w-3 h-3 fill-current" />
                    </motion.div>
                  )}
                </motion.div>
              )
            })}
          </div>

          {/* Progress bar */}
          <div className="mt-4 bg-gray-200 rounded-full h-2 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-600"
              initial={{ width: '0%' }}
              animate={{ width: `${(currentStep / (trainingSteps.length - 1)) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <div className="text-xs text-gray-600 text-center mt-1">
            Step {currentStep + 1} / {trainingSteps.length}
          </div>
        </div>

        {/* Right: Hook Details & Code */}
        <div className="space-y-4">
          {/* Hook Reference */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200 max-h-[280px] overflow-y-auto">
            <h4 className="font-bold text-gray-700 mb-3">Hook Reference</h4>
            <div className="space-y-2">
              {hooks.map(hook => (
                <div
                  key={hook.name}
                  className={`p-2 rounded-lg border transition-all ${
                    triggeredHooks.has(hook.name)
                      ? 'bg-gradient-to-r ' + hook.color + ' bg-opacity-20 border-blue-300'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {triggeredHooks.has(hook.name) ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <Circle className="w-4 h-4 text-gray-400" />
                    )}
                    <span className="font-mono text-sm font-semibold text-gray-800">
                      {hook.displayName}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 ml-6 mt-1">{hook.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Generated Code */}
          <div className="bg-gray-900 rounded-lg p-4 border-2 border-gray-700 max-h-[280px] overflow-hidden flex flex-col">
            <h4 className="font-bold text-gray-200 mb-3 flex items-center gap-2">
              <Code className="w-5 h-5 text-green-400" />
              Example Callback
            </h4>
            <pre className="flex-1 overflow-y-auto text-xs text-green-400 font-mono bg-gray-950 rounded p-3 border border-gray-700">
              {generateCode()}
            </pre>
          </div>
        </div>
      </div>

      {/* Execution History */}
      <div className="mt-6 bg-white rounded-lg p-4 border-2 border-gray-200">
        <h4 className="font-bold text-gray-700 mb-3">Execution History</h4>
        <div className="flex flex-wrap gap-2">
          <AnimatePresence>
            {stepHistory.map((step, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-semibold"
              >
                {idx + 1}. {step}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
        {stepHistory.length === 0 && (
          <p className="text-gray-400 text-sm">No steps executed yet. Click &quot;Start&quot; to begin.</p>
        )}
      </div>

      {/* Usage Tips */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-blue-800 mb-2">💡 Usage Tips</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• <strong>on_train_begin</strong>: Initialize custom state, log hyperparameters</li>
          <li>• <strong>on_step_end</strong>: Monitor gradients, implement gradient clipping</li>
          <li>• <strong>on_evaluate</strong>: Implement early stopping, custom metrics</li>
          <li>• <strong>on_save</strong>: Save additional model artifacts (tokenizer, config)</li>
          <li>• Use <code className="bg-blue-200 px-1 rounded">control.should_training_stop = True</code> to halt training</li>
        </ul>
      </div>
    </div>
  )
}

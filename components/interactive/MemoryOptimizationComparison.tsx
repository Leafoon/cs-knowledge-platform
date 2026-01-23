'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Zap } from 'lucide-react'

interface MethodMetrics {
  name: string
  memory: number
  speed: number
  accuracy: number
  color: string
}

export default function MemoryOptimizationComparison() {
  const [selectedTechnique, setSelectedTechnique] = useState<string>('none')

  const techniques: Record<string, MethodMetrics> = {
    none: {
      name: '基础训练',
      memory: 100,
      speed: 100,
      accuracy: 100,
      color: 'slate'
    },
    qlora: {
      name: 'QLoRA (4-bit)',
      memory: 23,
      speed: 85,
      accuracy: 98,
      color: 'purple'
    },
    'qlora+gc': {
      name: 'QLoRA + 梯度检查点',
      memory: 18,
      speed: 75,
      accuracy: 98,
      color: 'violet'
    },
    'qlora+gc+ga': {
      name: '+ 梯度累积',
      memory: 15,
      speed: 70,
      accuracy: 98,
      color: 'indigo'
    },
    'qlora+gc+ga+8bit': {
      name: '+ 8-bit Adam',
      memory: 12,
      speed: 68,
      accuracy: 98,
      color: 'blue'
    },
    full: {
      name: '全部优化',
      memory: 11,
      speed: 65,
      accuracy: 97,
      color: 'green'
    }
  }

  const selected = techniques[selectedTechnique]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-slate-900 dark:to-emerald-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-emerald-500" />
          显存优化技巧组合
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          逐步添加优化技术，在 6GB 显卡上训练 7B 模型
        </p>
      </div>

      {/* Technique Selector */}
      <div className="mb-6 space-y-2">
        {Object.entries(techniques).map(([key, tech]) => (
          <button
            key={key}
            onClick={() => setSelectedTechnique(key)}
            className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
              selectedTechnique === key
                ? `border-${tech.color}-500 bg-${tech.color}-50 dark:bg-${tech.color}-900/20`
                : 'border-slate-200 dark:border-slate-700 hover:border-slate-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className={`text-sm font-semibold text-${tech.color}-700 dark:text-${tech.color}-300`}>
                {tech.name}
              </span>
              <span className={`text-xs text-${tech.color}-600 dark:text-${tech.color}-400`}>
                {tech.memory}% 显存
              </span>
            </div>
          </button>
        ))}
      </div>

      {/* Metrics Bars */}
      <div className="space-y-4 mb-6">
        {/* Memory */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-700 dark:text-slate-300 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-500" />
              显存占用
            </span>
            <span className={`font-bold text-${selected.color}-600 dark:text-${selected.color}-400`}>
              {(selected.memory * 0.56).toFixed(1)} GB / 56 GB
            </span>
          </div>
          <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${selected.memory}%` }}
              transition={{ duration: 0.5 }}
              className={`h-full bg-${selected.color}-500 flex items-center justify-end pr-3`}
            >
              <span className="text-xs font-bold text-white">
                {selected.memory}%
              </span>
            </motion.div>
          </div>
        </div>

        {/* Speed */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-700 dark:text-slate-300 flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-500" />
              训练速度
            </span>
            <span className={`font-bold text-${selected.color}-600 dark:text-${selected.color}-400`}>
              {selected.speed}%
            </span>
          </div>
          <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${selected.speed}%` }}
              transition={{ duration: 0.5 }}
              className="h-full bg-orange-500 flex items-center justify-end pr-3"
            >
              <span className="text-xs font-bold text-white">
                {selected.speed}%
              </span>
            </motion.div>
          </div>
        </div>

        {/* Accuracy */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-700 dark:text-slate-300">模型性能</span>
            <span className={`font-bold text-${selected.color}-600 dark:text-${selected.color}-400`}>
              {selected.accuracy}%
            </span>
          </div>
          <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${selected.accuracy}%` }}
              transition={{ duration: 0.5 }}
              className="h-full bg-green-500 flex items-center justify-end pr-3"
            >
              <span className="text-xs font-bold text-white">
                {selected.accuracy}%
              </span>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Technique Details */}
      <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 mb-6">
        <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          应用的优化技术：
        </div>
        <div className="space-y-2 text-sm">
          {selectedTechnique !== 'none' && (
            <div className="flex items-start gap-2">
              <span className="text-purple-500">✓</span>
              <span className="text-slate-600 dark:text-slate-400">
                <strong>QLoRA (4-bit):</strong> 模型权重量化到 4-bit，显存 -77%
              </span>
            </div>
          )}
          {['qlora+gc', 'qlora+gc+ga', 'qlora+gc+ga+8bit', 'full'].includes(selectedTechnique) && (
            <div className="flex items-start gap-2">
              <span className="text-violet-500">✓</span>
              <span className="text-slate-600 dark:text-slate-400">
                <strong>梯度检查点:</strong> 重计算激活值，显存 -30%，速度 -15%
              </span>
            </div>
          )}
          {['qlora+gc+ga', 'qlora+gc+ga+8bit', 'full'].includes(selectedTechnique) && (
            <div className="flex items-start gap-2">
              <span className="text-indigo-500">✓</span>
              <span className="text-slate-600 dark:text-slate-400">
                <strong>梯度累积:</strong> batch_size=1 累积16步，显存 -20%
              </span>
            </div>
          )}
          {['qlora+gc+ga+8bit', 'full'].includes(selectedTechnique) && (
            <div className="flex items-start gap-2">
              <span className="text-blue-500">✓</span>
              <span className="text-slate-600 dark:text-slate-400">
                <strong>8-bit Adam:</strong> 优化器状态量化，显存 -25%
              </span>
            </div>
          )}
          {selectedTechnique === 'full' && (
            <div className="flex items-start gap-2">
              <span className="text-green-500">✓</span>
              <span className="text-slate-600 dark:text-slate-400">
                <strong>CPU Offload:</strong> 部分权重卸载到 CPU，显存 -10%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">完整优化配置</div>
        <div className="font-mono text-xs text-green-400 space-y-1">
          <div># 1. QLoRA 4-bit</div>
          <div className="text-slate-300">bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)</div>
          <div className="mt-2"># 2. 梯度检查点</div>
          <div className="text-slate-300">model.gradient_checkpointing_enable()</div>
          <div className="mt-2"># 3. 梯度累积 + 8-bit Adam</div>
          <div className="text-slate-300">TrainingArguments(</div>
          <div className="text-slate-300 ml-4">per_device_train_batch_size=1,</div>
          <div className="text-slate-300 ml-4">gradient_accumulation_steps=16,</div>
          <div className="text-slate-300 ml-4">optim=&quot;paged_adamw_8bit&quot;</div>
          <div className="text-slate-300">)</div>
          <div className="mt-2 text-yellow-400"># 结果: 7B 模型在 6GB 显卡上训练！</div>
        </div>
      </div>
    </div>
  )
}

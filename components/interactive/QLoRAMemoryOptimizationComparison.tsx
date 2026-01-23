'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { MemoryStick, Zap, TrendingDown } from 'lucide-react'

export default function QLoRAMemoryOptimizationComparison() {
  const configurations = [
    {
      name: '基础 QLoRA',
      settings: 'NF4 + r=16 + bs=4',
      memory: 9.2,
      speed: 15,
      color: 'blue',
    },
    {
      name: '+ Gradient Checkpointing',
      settings: 'gradient_checkpointing=True',
      memory: 6.8,
      speed: 12,
      color: 'green',
    },
    {
      name: '+ 小 Batch Size',
      settings: 'bs=2 + grad_accum=8',
      memory: 5.5,
      speed: 11,
      color: 'purple',
    },
    {
      name: '+ Paged AdamW',
      settings: 'optim="paged_adamw_8bit"',
      memory: 4.8,
      speed: 10.5,
      color: 'orange',
    },
  ]

  const maxMemory = Math.max(...configurations.map(c => c.memory))

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <MemoryStick className="w-8 h-8 text-green-600" />
        <h3 className="text-2xl font-bold text-slate-800">QLoRA 显存优化策略对比</h3>
      </div>

      {/* 对比图表 */}
      <div className="space-y-4 mb-6">
        {configurations.map((config, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.15 }}
            className="bg-white p-5 rounded-lg shadow"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h4 className={`text-lg font-bold text-${config.color}-800`}>{config.name}</h4>
                <div className="text-sm text-slate-600 font-mono">{config.settings}</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-600">显存</div>
                <div className={`text-2xl font-bold text-${config.color}-600`}>
                  {config.memory} GB
                </div>
              </div>
            </div>

            {/* 显存条 */}
            <div className="mb-3">
              <div className="text-xs text-slate-600 mb-1">显存占用</div>
              <div className="relative h-8 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(config.memory / maxMemory) * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.15 + 0.2 }}
                  className={`h-full bg-gradient-to-r from-${config.color}-400 to-${config.color}-600 flex items-center justify-end px-3`}
                >
                  <span className="text-white text-xs font-bold">
                    {((config.memory / 24) * 100).toFixed(0)}% of 24GB
                  </span>
                </motion.div>
              </div>
            </div>

            {/* 速度条 */}
            <div>
              <div className="text-xs text-slate-600 mb-1">训练速度</div>
              <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(config.speed / 15) * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.15 + 0.3 }}
                  className={`h-full bg-${config.color}-300`}
                />
                <div className="absolute inset-0 flex items-center px-3 text-xs font-bold text-slate-700">
                  {config.speed} it/s
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* 详细对比表 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">策略详解</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded">
            <div className="font-bold text-green-800 mb-2 flex items-center gap-2">
              <TrendingDown className="w-4 h-4" />
              Gradient Checkpointing
            </div>
            <div className="text-sm text-slate-700 space-y-1">
              <div>• 节省 <strong>30-40%</strong> 显存</div>
              <div>• 训练速度降低 <strong>20%</strong></div>
              <div>• 反向传播时重算激活值</div>
            </div>
          </div>

          <div className="p-4 bg-purple-50 border border-purple-200 rounded">
            <div className="font-bold text-purple-800 mb-2">小 Batch + 梯度累积</div>
            <div className="text-sm text-slate-700 space-y-1">
              <div>• 减少单步激活值</div>
              <div>• 保持等效 batch size</div>
              <div>• 灵活控制显存</div>
            </div>
          </div>

          <div className="p-4 bg-orange-50 border border-orange-200 rounded">
            <div className="font-bold text-orange-800 mb-2 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Paged AdamW
            </div>
            <div className="text-sm text-slate-700 space-y-1">
              <div>• 优化器状态 CPU offload</div>
              <div>• 避免峰值 OOM</div>
              <div>• 速度影响 <strong>{'<5%'}</strong></div>
            </div>
          </div>

          <div className="p-4 bg-blue-50 border border-blue-200 rounded">
            <div className="font-bold text-blue-800 mb-2">组合策略</div>
            <div className="text-sm text-slate-700 space-y-1">
              <div>• 多策略叠加效果显著</div>
              <div>• RTX 4090 可训练 13B 模型</div>
              <div>• 性能损失可接受</div>
            </div>
          </div>
        </div>
      </div>

      {/* 极限挑战 */}
      <div className="mt-6 p-5 bg-gradient-to-r from-red-50 to-orange-50 border-2 border-red-300 rounded-lg">
        <h4 className="font-bold text-red-800 mb-3">🔥 极限挑战：70B 模型单卡训练</h4>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-xs text-slate-600 mb-1">配置</div>
            <div className="text-sm text-slate-800">
              • NF4 + 双重量化<br />
              • r=8, 仅 Q/V<br />
              • bs=1, grad_accum=16<br />
              • Gradient checkpointing<br />
              • Paged AdamW<br />
              • max_length=512
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-600 mb-1">显存占用</div>
            <div className="text-3xl font-bold text-red-600">23.5 GB</div>
            <div className="text-xs text-slate-500 mt-1">RTX 4090 可用！</div>
          </div>
          <div>
            <div className="text-xs text-slate-600 mb-1">训练速度</div>
            <div className="text-3xl font-bold text-orange-600">2.5 it/s</div>
            <div className="text-xs text-slate-500 mt-1">可接受</div>
          </div>
        </div>
      </div>
    </div>
  )
}

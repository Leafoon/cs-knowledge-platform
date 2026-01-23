'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Server, AlertCircle, CheckCircle, XCircle } from 'lucide-react'

interface ModelConfig {
  name: string
  params: string
  paramsNum: number
  fp16Memory: number
  optimizerMemory: number
  gradientMemory: number
  activationMemory: number
  canTrainSingle: boolean
}

const models: ModelConfig[] = [
  {
    name: 'BERT-base',
    params: '110M',
    paramsNum: 110_000_000,
    fp16Memory: 0.2,
    optimizerMemory: 0.9,
    gradientMemory: 0.2,
    activationMemory: 1.5,
    canTrainSingle: true,
  },
  {
    name: 'GPT-2',
    params: '1.5B',
    paramsNum: 1_500_000_000,
    fp16Memory: 3,
    optimizerMemory: 12,
    gradientMemory: 3,
    activationMemory: 5,
    canTrainSingle: true,
  },
  {
    name: 'LLaMA-7B',
    params: '7B',
    paramsNum: 7_000_000_000,
    fp16Memory: 14,
    optimizerMemory: 56,
    gradientMemory: 14,
    activationMemory: 10,
    canTrainSingle: false,
  },
  {
    name: 'LLaMA-13B',
    params: '13B',
    paramsNum: 13_000_000_000,
    fp16Memory: 26,
    optimizerMemory: 104,
    gradientMemory: 26,
    activationMemory: 18,
    canTrainSingle: false,
  },
  {
    name: 'LLaMA-70B',
    params: '70B',
    paramsNum: 70_000_000_000,
    fp16Memory: 140,
    optimizerMemory: 560,
    gradientMemory: 140,
    activationMemory: 80,
    canTrainSingle: false,
  },
]

export default function DistributedTrainingNeedVisualizer() {
  const [selectedModel, setSelectedModel] = useState<ModelConfig>(models[2])

  const totalMemory = selectedModel.fp16Memory + selectedModel.optimizerMemory + 
                      selectedModel.gradientMemory + selectedModel.activationMemory

  const a100Capacity = 80 // GB

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Server className="w-8 h-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-slate-800">分布式训练显存需求分析</h3>
      </div>

      {/* 模型选择器 */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-3">
          选择模型规模
        </label>
        <div className="grid grid-cols-5 gap-3">
          {models.map((model) => (
            <button
              key={model.name}
              onClick={() => setSelectedModel(model)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedModel.name === model.name
                  ? 'border-blue-600 bg-blue-50 shadow-md'
                  : 'border-slate-200 bg-white hover:border-blue-300'
              }`}
            >
              <div className="text-center">
                <div className="font-bold text-slate-800">{model.name}</div>
                <div className="text-sm text-slate-500">{model.params}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 显存需求详情 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 左侧：显存组成 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-bold text-lg text-slate-800 mb-4">显存需求分解</h4>
          
          <div className="space-y-3">
            {[
              { label: '模型参数 (FP16)', value: selectedModel.fp16Memory, color: 'bg-blue-500' },
              { label: '优化器状态 (Adam)', value: selectedModel.optimizerMemory, color: 'bg-purple-500' },
              { label: '梯度', value: selectedModel.gradientMemory, color: 'bg-green-500' },
              { label: '激活值', value: selectedModel.activationMemory, color: 'bg-orange-500' },
            ].map((item, idx) => (
              <div key={idx}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-700">{item.label}</span>
                  <span className="font-mono font-bold text-slate-800">
                    {item.value.toFixed(1)} GB
                  </span>
                </div>
                <div className="relative h-6 bg-slate-100 rounded overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(item.value / totalMemory) * 100}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                    className={`h-full ${item.color} flex items-center justify-end px-2`}
                  >
                    <span className="text-xs text-white font-medium">
                      {((item.value / totalMemory) * 100).toFixed(0)}%
                    </span>
                  </motion.div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-200">
            <div className="flex justify-between items-center">
              <span className="font-bold text-slate-800">总计</span>
              <span className="font-mono text-2xl font-bold text-blue-600">
                {totalMemory.toFixed(1)} GB
              </span>
            </div>
          </div>
        </div>

        {/* 右侧：A100容量对比 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-bold text-lg text-slate-800 mb-4">A100 80GB 容量对比</h4>
          
          <div className="mb-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-700">显存需求 / GPU容量</span>
              <span className="font-mono font-bold text-slate-800">
                {totalMemory.toFixed(1)} / {a100Capacity} GB
              </span>
            </div>
            <div className="relative h-12 bg-slate-200 rounded-lg overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((totalMemory / a100Capacity) * 100, 100)}%` }}
                transition={{ duration: 1 }}
                className={`h-full ${
                  totalMemory <= a100Capacity ? 'bg-green-500' : 'bg-red-500'
                } flex items-center justify-center`}
              >
                <span className="text-white font-bold">
                  {((totalMemory / a100Capacity) * 100).toFixed(0)}%
                </span>
              </motion.div>
              {totalMemory > a100Capacity && (
                <div className="absolute inset-0 border-2 border-red-500 rounded-lg" />
              )}
            </div>
          </div>

          {/* 可训练性判断 */}
          <div className={`p-4 rounded-lg ${
            totalMemory <= a100Capacity 
              ? 'bg-green-50 border-2 border-green-200' 
              : 'bg-red-50 border-2 border-red-200'
          }`}>
            <div className="flex items-start gap-3">
              {totalMemory <= a100Capacity ? (
                <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <div className={`font-bold mb-1 ${
                  totalMemory <= a100Capacity ? 'text-green-800' : 'text-red-800'
                }`}>
                  {totalMemory <= a100Capacity ? '✓ 单卡可训练' : '✗ 单卡不足'}
                </div>
                <div className="text-sm text-slate-600">
                  {totalMemory <= a100Capacity ? (
                    <span>可在单张 A100 80GB 上训练（需适当优化）</span>
                  ) : (
                    <span>
                      需要 <strong className="text-red-700">
                        {Math.ceil(totalMemory / a100Capacity)} 张卡
                      </strong> 或使用分布式训练策略
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* 推荐策略 */}
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-blue-800 mb-1">推荐训练策略</div>
                <div className="text-sm text-slate-700">
                  {totalMemory <= 30 ? (
                    <span>单卡 DDP + 混合精度</span>
                  ) : totalMemory <= 100 ? (
                    <span>FSDP 或 DeepSpeed ZeRO-2</span>
                  ) : (
                    <span>DeepSpeed ZeRO-3 + CPU Offload</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 计算公式 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-2">显存计算公式</h4>
        <div className="font-mono text-sm text-slate-600 bg-slate-50 p-3 rounded border border-slate-200">
          <div>模型参数 = {selectedModel.paramsNum.toLocaleString()} × 2 bytes (FP16) = {selectedModel.fp16Memory} GB</div>
          <div>优化器状态 = {selectedModel.paramsNum.toLocaleString()} × 8 bytes (Adam FP32) = {selectedModel.optimizerMemory} GB</div>
          <div>梯度 = {selectedModel.paramsNum.toLocaleString()} × 2 bytes = {selectedModel.gradientMemory} GB</div>
          <div>激活值 ≈ batch_size × seq_len × hidden × layers × 4 ≈ {selectedModel.activationMemory} GB</div>
          <div className="mt-2 pt-2 border-t border-slate-300 font-bold">
            总计 = {totalMemory.toFixed(1)} GB
          </div>
        </div>
      </div>
    </div>
  )
}

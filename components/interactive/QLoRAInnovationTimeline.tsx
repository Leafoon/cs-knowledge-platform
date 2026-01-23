'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, TrendingDown, Zap, Database } from 'lucide-react'

interface Innovation {
  id: number
  year: string
  title: string
  description: string
  impact: string
  memoryReduction: string
  icon: React.ReactNode
  color: string
}

export default function QLoRAInnovationTimeline() {
  const [selectedInnovation, setSelectedInnovation] = useState(0)

  const innovations: Innovation[] = [
    {
      id: 0,
      year: '2021',
      title: 'LoRA (Low-Rank Adaptation)',
      description: '通过低秩分解 W = W₀ + BA 实现参数高效微调',
      impact: '可训练参数减少到 0.1-1%',
      memoryReduction: '50%',
      icon: <Sparkles className="w-6 h-6" />,
      color: 'blue',
    },
    {
      id: 1,
      year: '2023.05',
      title: 'QLoRA 论文发布',
      description: 'Efficient Finetuning of Quantized LLMs (Dettmers et al.)',
      impact: '首次实现 65B 模型单卡微调',
      memoryReduction: '75%',
      icon: <Database className="w-6 h-6" />,
      color: 'green',
    },
    {
      id: 2,
      year: '2023.05',
      title: '4-bit NormalFloat (NF4)',
      description: '专为正态分布权重设计的新数据类型，使用分位数量化',
      impact: '量化误差降低 30%（vs INT4）',
      memoryReduction: '4x vs FP16',
      icon: <TrendingDown className="w-6 h-6" />,
      color: 'purple',
    },
    {
      id: 3,
      year: '2023.05',
      title: '双重量化 (Double Quantization)',
      description: '量化权重的量化常数，节省额外 0.37 bits/param',
      impact: 'LLaMA-65B 额外节省 3GB',
      memoryReduction: '额外 8%',
      icon: <Database className="w-6 h-6" />,
      color: 'orange',
    },
    {
      id: 4,
      year: '2023.05',
      title: 'Paged Optimizers',
      description: '借鉴虚拟内存，自动 CPU-GPU 交换优化器状态',
      impact: '避免显存峰值 OOM',
      memoryReduction: '峰值降低 15%',
      icon: <Zap className="w-6 h-6" />,
      color: 'red',
    },
  ]

  const current = innovations[selectedInnovation]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Sparkles className="w-8 h-8 text-indigo-600" />
        <h3 className="text-2xl font-bold text-slate-800">QLoRA 创新历程</h3>
      </div>

      {/* 时间轴 */}
      <div className="relative mb-8">
        <div className="absolute top-10 left-0 w-full h-1 bg-slate-200" />
        <div className="relative flex justify-between items-start">
          {innovations.map((innovation, idx) => (
            <div key={innovation.id} className="flex flex-col items-center flex-1">
              <button
                onClick={() => setSelectedInnovation(idx)}
                className={`relative z-10 w-20 h-20 rounded-full border-4 transition-all duration-300 flex items-center justify-center ${
                  selectedInnovation === idx
                    ? `border-${innovation.color}-600 bg-${innovation.color}-100 shadow-lg scale-110`
                    : 'border-slate-300 bg-white hover:border-slate-400'
                }`}
              >
                <div className={`${selectedInnovation === idx ? `text-${innovation.color}-600` : 'text-slate-400'}`}>
                  {innovation.icon}
                </div>
              </button>
              <div className="mt-3 text-center">
                <div className={`text-sm font-bold ${
                  selectedInnovation === idx ? `text-${innovation.color}-600` : 'text-slate-600'
                }`}>
                  {innovation.year}
                </div>
                <div className="text-xs text-slate-500 mt-1 max-w-[120px]">
                  {innovation.title.split(' ')[0]}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedInnovation}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={`bg-white p-6 rounded-lg shadow-lg border-2 border-${current.color}-200`}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-xs text-slate-500 mb-1">{current.year}</div>
            <h4 className={`text-2xl font-bold text-${current.color}-800 mb-2`}>
              {current.title}
            </h4>
            <p className="text-slate-700">{current.description}</p>
          </div>
          <div className={`text-${current.color}-600`}>
            {current.icon}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mt-6">
          <div className={`p-4 bg-${current.color}-50 rounded-lg border border-${current.color}-200`}>
            <div className="text-sm text-slate-600 mb-1">核心影响</div>
            <div className={`text-lg font-bold text-${current.color}-800`}>{current.impact}</div>
          </div>
          <div className={`p-4 bg-${current.color}-50 rounded-lg border border-${current.color}-200`}>
            <div className="text-sm text-slate-600 mb-1">显存优化</div>
            <div className={`text-lg font-bold text-${current.color}-800`}>{current.memoryReduction}</div>
          </div>
        </div>

        {/* 特殊说明 */}
        {selectedInnovation === 1 && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="font-bold text-green-800 mb-2">🎯 突破性成果</div>
            <div className="text-sm text-slate-700">
              QLoRA 论文在 Hugging Face + University of Washington 合作下发布，首次证明：
              <ul className="list-disc ml-5 mt-2 space-y-1">
                <li>65B 模型可在单张 48GB GPU 微调（A6000）</li>
                <li>性能与全精度微调相当（MMLU: 46.8% vs 47.1%）</li>
                <li>LoRA 权重仅 ~80MB（vs 130GB 全模型）</li>
              </ul>
            </div>
          </div>
        )}

        {selectedInnovation === 2 && (
          <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-lg">
            <div className="font-bold text-purple-800 mb-2">📊 NF4 vs INT4 对比</div>
            <div className="text-sm text-slate-700">
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="p-2 bg-white rounded">
                  <div className="text-xs text-slate-500">INT4 均匀分布</div>
                  <div className="font-mono text-xs">[-8, -7, ..., 6, 7]</div>
                </div>
                <div className="p-2 bg-purple-100 rounded">
                  <div className="text-xs text-slate-500">NF4 分位数分布</div>
                  <div className="font-mono text-xs">[-1.0, -0.69, ..., 1.0]</div>
                </div>
              </div>
              <div className="mt-2 text-xs">
                NF4 对神经网络权重（正态分布）量化误差降低 <strong>30%</strong>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* 综合对比 */}
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">显存占用对比（LLaMA-65B）</h4>
        <div className="space-y-3">
          {[
            { label: 'FP32 全精度', memory: 260, color: 'slate', percent: 100 },
            { label: 'FP16 全精度', memory: 130, color: 'blue', percent: 50 },
            { label: 'LoRA (FP16)', memory: 65, color: 'green', percent: 25 },
            { label: 'QLoRA (NF4)', memory: 32.5, color: 'purple', percent: 12.5 },
          ].map((item) => (
            <div key={item.label}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-700">{item.label}</span>
                <span className="font-bold">{item.memory} GB</span>
              </div>
              <div className="relative h-8 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${item.percent}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className={`h-full bg-gradient-to-r from-${item.color}-400 to-${item.color}-600 flex items-center justify-end px-3`}
                >
                  <span className="text-white text-xs font-bold">{item.percent}%</span>
                </motion.div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

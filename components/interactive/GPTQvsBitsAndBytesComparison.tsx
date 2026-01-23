'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Clock, Target, Cpu, Database } from 'lucide-react'

export default function GPTQvsBitsAndBytesComparison() {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)

  const comparison = {
    algorithm: {
      label: '量化算法',
      icon: Cpu,
      gptq: { value: 'Optimal Brain Quantization', color: 'blue' },
      bnb: { value: 'NormalFloat + 双重量化', color: 'purple' },
      detail: 'GPTQ 使用 Hessian 矩阵二阶信息优化，bitsandbytes 使用 NF4 编码',
    },
    calibration: {
      label: '校准数据',
      icon: Database,
      gptq: { value: '✓ 需要 (128-256 samples)', color: 'orange' },
      bnb: { value: '✗ 零校准', color: 'green' },
      detail: 'GPTQ 需要校准数据计算激活值统计，bitsandbytes 直接量化权重',
    },
    time: {
      label: '量化时间',
      icon: Clock,
      gptq: { value: '5-10 分钟', color: 'red' },
      bnb: { value: '< 1 分钟', color: 'green' },
      detail: 'GPTQ 需要逐层量化并计算 Hessian，bitsandbytes 加载时自动量化',
    },
    speed: {
      label: '推理速度',
      icon: Zap,
      gptq: { value: '35 tokens/s (更快)', color: 'green' },
      bnb: { value: '28 tokens/s', color: 'blue' },
      detail: 'GPTQ 有专门优化的 CUDA kernel，推理速度更快',
    },
    memory: {
      label: '显存占用',
      icon: Database,
      gptq: { value: '4.5 GB', color: 'blue' },
      bnb: { value: '4.8 GB', color: 'blue' },
      detail: '两者显存占用接近，bitsandbytes 略高因为 paged optimizer',
    },
    accuracy: {
      label: '精度 (PPL)',
      icon: Target,
      gptq: { value: '6.12 (更高)', color: 'green' },
      bnb: { value: '6.28', color: 'yellow' },
      detail: 'GPTQ 基于二阶优化，精度更高（PPL 越低越好）',
    },
    finetune: {
      label: '微调支持',
      icon: Zap,
      gptq: { value: '✗ 困难 (需解量化)', color: 'red' },
      bnb: { value: '✓ 原生支持 (QLoRA)', color: 'green' },
      detail: 'bitsandbytes 专为 QLoRA 设计，支持量化模型微调',
    },
  }

  const metrics = Object.entries(comparison)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        GPTQ vs bitsandbytes 对比
      </h3>

      {/* 对比表格 */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden mb-6">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-slate-100 to-slate-200 border-b-2 border-slate-300">
              <th className="text-left py-4 px-6 font-bold text-slate-700">指标</th>
              <th className="text-center py-4 px-6 font-bold text-blue-600">GPTQ</th>
              <th className="text-center py-4 px-6 font-bold text-purple-600">bitsandbytes</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(([key, metric], idx) => {
              const Icon = metric.icon
              const isSelected = selectedMetric === key
              return (
                <motion.tr
                  key={key}
                  onClick={() => setSelectedMetric(isSelected ? null : key)}
                  className={`border-b border-slate-100 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-50' : 'hover:bg-slate-50'
                  }`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <td className="py-4 px-6">
                    <div className="flex items-center gap-3">
                      <Icon className="w-5 h-5 text-slate-500" />
                      <span className="font-medium text-slate-800">{metric.label}</span>
                    </div>
                  </td>
                  <td className="py-4 px-6 text-center">
                    <div className={`inline-block px-3 py-1 rounded-full bg-${metric.gptq.color}-100 text-${metric.gptq.color}-700 text-sm font-medium`}>
                      {metric.gptq.value}
                    </div>
                  </td>
                  <td className="py-4 px-6 text-center">
                    <div className={`inline-block px-3 py-1 rounded-full bg-${metric.bnb.color}-100 text-${metric.bnb.color}-700 text-sm font-medium`}>
                      {metric.bnb.value}
                    </div>
                  </td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* 详细说明 */}
      {selectedMetric && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200 mb-6"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center">
              {React.createElement(comparison[selectedMetric as keyof typeof comparison].icon, {
                className: 'w-5 h-5 text-white',
              })}
            </div>
            <h4 className="font-bold text-lg text-slate-800">
              {comparison[selectedMetric as keyof typeof comparison].label}
            </h4>
          </div>
          <p className="text-slate-700">
            {comparison[selectedMetric as keyof typeof comparison].detail}
          </p>
        </motion.div>
      )}

      {/* 性能基准 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <h4 className="font-bold text-slate-800 mb-4">性能基准 (LLaMA-7B on A100)</h4>
        
        <div className="space-y-4">
          {/* 困惑度 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-slate-600">困惑度 (越低越好)</span>
              <span className="text-xs text-slate-500">FP16 baseline: 5.68</span>
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-blue-600 font-medium">GPTQ</span>
                  <span className="text-slate-600">6.12 (+0.44)</span>
                </div>
                <div className="h-6 bg-blue-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '92.8%' }}
                    transition={{ duration: 1, delay: 0.2 }}
                  >
                    <span className="text-xs text-white font-bold">92.8%</span>
                  </motion.div>
                </div>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-purple-600 font-medium">bitsandbytes</span>
                  <span className="text-slate-600">6.28 (+0.60)</span>
                </div>
                <div className="h-6 bg-purple-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '90.4%' }}
                    transition={{ duration: 1, delay: 0.4 }}
                  >
                    <span className="text-xs text-white font-bold">90.4%</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>

          {/* 推理速度 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-slate-600">推理速度 (越高越好)</span>
              <span className="text-xs text-slate-500">FP16: 18 tokens/s</span>
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-blue-600 font-medium">GPTQ</span>
                  <span className="text-slate-600">35 tokens/s (1.94x)</span>
                </div>
                <div className="h-6 bg-blue-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 1, delay: 0.6 }}
                  >
                    <span className="text-xs text-white font-bold">1.94x</span>
                  </motion.div>
                </div>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-purple-600 font-medium">bitsandbytes</span>
                  <span className="text-slate-600">28 tokens/s (1.56x)</span>
                </div>
                <div className="h-6 bg-purple-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '80%' }}
                    transition={{ duration: 1, delay: 0.8 }}
                  >
                    <span className="text-xs text-white font-bold">1.56x</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>

          {/* 显存占用 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-slate-600">显存占用 (越低越好)</span>
              <span className="text-xs text-slate-500">FP16: 14 GB</span>
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-blue-600 font-medium">GPTQ</span>
                  <span className="text-slate-600">4.5 GB (68% ↓)</span>
                </div>
                <div className="h-6 bg-blue-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '32.1%' }}
                    transition={{ duration: 1, delay: 1.0 }}
                  >
                    <span className="text-xs text-white font-bold">4.5 GB</span>
                  </motion.div>
                </div>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-purple-600 font-medium">bitsandbytes</span>
                  <span className="text-slate-600">4.8 GB (66% ↓)</span>
                </div>
                <div className="h-6 bg-purple-100 rounded-lg overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-end px-2"
                    initial={{ width: 0 }}
                    animate={{ width: '34.3%' }}
                    transition={{ duration: 1, delay: 1.2 }}
                  >
                    <span className="text-xs text-white font-bold">4.8 GB</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 选择建议 */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
          <h4 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
            <Target className="w-5 h-5" />
            选择 GPTQ
          </h4>
          <ul className="text-sm text-blue-700 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-0.5">✓</span>
              <span><strong>纯推理部署</strong>（不需要微调）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-0.5">✓</span>
              <span><strong>追求速度</strong>（延迟敏感）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-0.5">✓</span>
              <span><strong>有校准数据</strong>（128+ samples）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-0.5">✓</span>
              <span><strong>精度优先</strong>（PPL 更低）</span>
            </li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl border border-purple-200">
          <h4 className="font-bold text-purple-800 mb-3 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            选择 bitsandbytes
          </h4>
          <ul className="text-sm text-purple-700 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-purple-500 mt-0.5">✓</span>
              <span><strong>需要微调</strong>（QLoRA 原生支持）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-500 mt-0.5">✓</span>
              <span><strong>无校准数据</strong>（零校准）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-500 mt-0.5">✓</span>
              <span><strong>快速实验</strong>（秒级量化）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-500 mt-0.5">✓</span>
              <span><strong>简单易用</strong>（一行代码）</span>
            </li>
          </ul>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 点击表格行查看详细说明
      </div>
    </div>
  )
}

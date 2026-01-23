'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, CheckCircle, XCircle, AlertCircle } from 'lucide-react'

interface Method {
  name: string
  params: string
  speed: 'fastest' | 'fast' | 'medium' | 'slow'
  performance: number
  memory: number
  description: string
  pros: string[]
  cons: string[]
  useCase: string
}

export default function PEFTMethodComparisonTable() {
  const [selectedMethod, setSelectedMethod] = useState('LoRA')

  const methods: Method[] = [
    {
      name: 'LoRA',
      params: '0.1-1%',
      speed: 'fast',
      performance: 99.5,
      memory: 50,
      description: '低秩分解，训练适配器矩阵 BA',
      pros: ['性能最佳', '生态成熟', '推理快'],
      cons: ['需调整 rank', '多任务切换慢'],
      useCase: '通用首选，微调大模型',
    },
    {
      name: 'QLoRA',
      params: '0.1-1%',
      speed: 'medium',
      performance: 99.2,
      memory: 75,
      description: 'LoRA + 4-bit 量化基座',
      pros: ['极低显存', '可训练超大模型', '性能接近 LoRA'],
      cons: ['需 bitsandbytes', '推理稍慢'],
      useCase: '显存受限（<24GB）',
    },
    {
      name: 'Prefix Tuning',
      params: '0.1-0.5%',
      speed: 'fastest',
      performance: 98,
      memory: 60,
      description: '输入前添加可训练 prefix 向量',
      pros: ['参数极少', '推理最快', '无额外矩阵乘'],
      cons: ['性能略低', 'prefix 占用输入长度'],
      useCase: '超大模型（>10B）',
    },
    {
      name: 'P-Tuning v2',
      params: '0.1-0.5%',
      speed: 'fast',
      performance: 99,
      memory: 60,
      description: '每层都添加 prefix（深度 prompt）',
      pros: ['性能优于 Prefix', 'T5/GLM 友好'],
      cons: ['参数稍多于 Prefix'],
      useCase: 'T5、GLM 系列模型',
    },
    {
      name: 'Prompt Tuning',
      params: '<0.01%',
      speed: 'fastest',
      performance: 97,
      memory: 70,
      description: '仅优化输入层 soft prompts',
      pros: ['参数最少', '速度最快'],
      cons: ['性能较低', '需大模型'],
      useCase: 'Few-shot 学习',
    },
    {
      name: 'Adapter',
      params: '1-5%',
      speed: 'slow',
      performance: 99,
      memory: 40,
      description: '插入 bottleneck 全连接层',
      pros: ['多任务切换快', '性能稳定'],
      cons: ['推理慢', '参数相对多'],
      useCase: '多任务学习',
    },
    {
      name: '(IA)³',
      params: '<0.01%',
      speed: 'fastest',
      performance: 98,
      memory: 65,
      description: '学习逐元素缩放向量',
      pros: ['参数最少', '推理无开销', '可融合'],
      cons: ['性能略低于 LoRA'],
      useCase: '极端资源受限',
    },
  ]

  const current = methods.find(m => m.name === selectedMethod) || methods[0]

  const getSpeedColor = (speed: Method['speed']) => {
    const colors = {
      fastest: 'green',
      fast: 'blue',
      medium: 'orange',
      slow: 'red',
    }
    return colors[speed]
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-violet-600" />
        <h3 className="text-2xl font-bold text-slate-800">PEFT 方法全面对比</h3>
      </div>

      {/* 方法选择 */}
      <div className="grid grid-cols-4 gap-2 mb-6">
        {methods.map((method) => (
          <button
            key={method.name}
            onClick={() => setSelectedMethod(method.name)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedMethod === method.name
                ? 'border-violet-600 bg-violet-50 shadow-md'
                : 'border-slate-200 bg-white hover:border-violet-300'
            }`}
          >
            <div className={`font-bold ${
              selectedMethod === method.name ? 'text-violet-900' : 'text-slate-700'
            }`}>
              {method.name}
            </div>
            <div className="text-xs text-slate-600 mt-1">{method.params}</div>
          </button>
        ))}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedMethod}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* 性能指标 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-4">{current.name} - 核心指标</h4>
          
          <div className="grid grid-cols-3 gap-6 mb-4">
            <div>
              <div className="text-sm text-slate-600 mb-2">推理速度</div>
              <div className={`text-2xl font-bold text-${getSpeedColor(current.speed)}-600 mb-2`}>
                {current.speed.toUpperCase()}
              </div>
              <div className="h-2 bg-slate-100 rounded-full">
                <div
                  className={`h-full bg-${getSpeedColor(current.speed)}-500 rounded-full`}
                  style={{
                    width: current.speed === 'fastest' ? '100%' :
                           current.speed === 'fast' ? '80%' :
                           current.speed === 'medium' ? '60%' : '40%'
                  }}
                />
              </div>
            </div>

            <div>
              <div className="text-sm text-slate-600 mb-2">性能（vs Full FT）</div>
              <div className="text-2xl font-bold text-blue-600 mb-2">
                {current.performance}%
              </div>
              <div className="h-2 bg-slate-100 rounded-full">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${current.performance}%` }}
                />
              </div>
            </div>

            <div>
              <div className="text-sm text-slate-600 mb-2">显存节省</div>
              <div className="text-2xl font-bold text-green-600 mb-2">
                {current.memory}%
              </div>
              <div className="h-2 bg-slate-100 rounded-full">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{ width: `${current.memory}%` }}
                />
              </div>
            </div>
          </div>

          <div className="p-3 bg-violet-50 border border-violet-200 rounded text-sm text-slate-700">
            <strong>核心原理</strong>: {current.description}
          </div>
        </div>

        {/* 优缺点 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <h4 className="font-bold text-green-800">优点</h4>
            </div>
            <ul className="space-y-2">
              {current.pros.map((pro, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>{pro}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <XCircle className="w-5 h-5 text-red-600" />
              <h4 className="font-bold text-red-800">缺点</h4>
            </div>
            <ul className="space-y-2">
              {current.cons.map((con, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-red-600 mt-0.5">✗</span>
                  <span>{con}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* 适用场景 */}
        <div className="bg-white p-5 rounded-lg shadow border-2 border-yellow-300">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            <h4 className="font-bold text-yellow-800">推荐使用场景</h4>
          </div>
          <div className="text-slate-700">{current.useCase}</div>
        </div>
      </motion.div>

      {/* 对比表格 */}
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">完整对比表</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-slate-300">
                <th className="text-left py-2 px-3">方法</th>
                <th className="text-center py-2 px-3">参数量</th>
                <th className="text-center py-2 px-3">速度</th>
                <th className="text-center py-2 px-3">性能</th>
                <th className="text-center py-2 px-3">显存节省</th>
              </tr>
            </thead>
            <tbody>
              {methods.map((method) => (
                <tr
                  key={method.name}
                  className={`border-b border-slate-200 ${
                    selectedMethod === method.name ? 'bg-violet-50' : ''
                  }`}
                >
                  <td className="py-2 px-3 font-bold">{method.name}</td>
                  <td className="py-2 px-3 text-center font-mono">{method.params}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-bold text-${getSpeedColor(method.speed)}-700 bg-${getSpeedColor(method.speed)}-100`}>
                      {method.speed}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center font-mono">{method.performance}%</td>
                  <td className="py-2 px-3 text-center font-mono">{method.memory}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

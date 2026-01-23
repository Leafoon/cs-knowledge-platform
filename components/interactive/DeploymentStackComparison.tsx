'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Server, Zap, TrendingUp, CheckCircle, XCircle } from 'lucide-react'

type Stack = 'transformers' | 'vllm' | 'tgi' | 'tensorrt'

interface StackConfig {
  id: Stack
  name: string
  description: string
  pros: string[]
  cons: string[]
  throughput: number
  latency: number
  memoryEfficiency: number
  easeOfUse: number
  color: string
}

export default function DeploymentStackComparison() {
  const [selectedStack, setSelectedStack] = useState<Stack>('transformers')

  const stacks: StackConfig[] = [
    {
      id: 'transformers',
      name: 'Transformers (基线)',
      description: 'Hugging Face 原生推理',
      pros: [
        '零配置，开箱即用',
        '支持所有模型架构',
        '与训练代码无缝衔接',
        '丰富的文档和社区',
      ],
      cons: [
        '吞吐量较低',
        '内存利用率不足',
        '批处理效率一般',
        '缺乏高级优化',
      ],
      throughput: 20,
      latency: 450,
      memoryEfficiency: 60,
      easeOfUse: 100,
      color: 'blue',
    },
    {
      id: 'vllm',
      name: 'vLLM',
      description: 'PagedAttention + Continuous Batching',
      pros: [
        'PagedAttention 内存优化',
        'Continuous Batching 高吞吐',
        'OpenAI 兼容 API',
        '自动并发调度',
      ],
      cons: [
        '仅支持主流模型',
        '需要 CUDA 11.8+',
        '配置稍复杂',
        '内存预算需调优',
      ],
      throughput: 68,
      latency: 180,
      memoryEfficiency: 90,
      easeOfUse: 75,
      color: 'green',
    },
    {
      id: 'tgi',
      name: 'TGI (Text Generation Inference)',
      description: 'Hugging Face 官方推理框架',
      pros: [
        'Flash Attention 2 自动启用',
        '张量并行支持',
        'Safetensors 快速加载',
        'Docker 一键部署',
      ],
      cons: [
        '主要针对文本生成',
        'Docker 镜像较大',
        '需要配置端口映射',
        '量化选项有限',
      ],
      throughput: 65,
      latency: 190,
      memoryEfficiency: 85,
      easeOfUse: 80,
      color: 'purple',
    },
    {
      id: 'tensorrt',
      name: 'TensorRT-LLM',
      description: 'NVIDIA 硬件加速推理',
      pros: [
        '最快推理速度（2-4x）',
        'Kernel融合优化',
        'INT8/FP8 量化支持',
        'NVIDIA GPU 最优',
      ],
      cons: [
        '配置复杂（需编译）',
        '仅支持 NVIDIA GPU',
        '模型兼容性有限',
        '调试困难',
      ],
      throughput: 95,
      latency: 120,
      memoryEfficiency: 95,
      easeOfUse: 40,
      color: 'orange',
    },
  ]

  const currentStack = stacks.find(s => s.id === selectedStack)!

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Server className="w-8 h-8 text-indigo-600" />
        <h3 className="text-2xl font-bold text-slate-800">推理部署方案对比</h3>
      </div>

      {/* 方案选择器 */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {stacks.map((stack) => (
          <button
            key={stack.id}
            onClick={() => setSelectedStack(stack.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedStack === stack.id
                ? `border-${stack.color}-600 bg-${stack.color}-50 shadow-md`
                : 'border-slate-200 bg-white hover:border-indigo-300'
            }`}
          >
            <div className={`font-bold mb-1 ${
              selectedStack === stack.id ? `text-${stack.color}-900` : 'text-slate-700'
            }`}>
              {stack.name}
            </div>
            <div className="text-xs text-slate-600">{stack.description}</div>
          </button>
        ))}
      </div>

      {/* 详细信息 */}
      <motion.div
        key={selectedStack}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-6"
      >
        {/* 性能指标 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-indigo-600" />
            性能指标
          </h4>

          <div className="space-y-4">
            {/* 吞吐量 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-medium">吞吐量 (requests/s)</span>
                <span className="font-mono font-bold text-slate-800">{currentStack.throughput}</span>
              </div>
              <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(currentStack.throughput / 100) * 100}%` }}
                  transition={{ duration: 0.8 }}
                  className={`h-full bg-gradient-to-r from-${currentStack.color}-400 to-${currentStack.color}-600`}
                />
              </div>
            </div>

            {/* 延迟 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-medium">延迟 (P95, ms)</span>
                <span className="font-mono font-bold text-slate-800">{currentStack.latency}</span>
              </div>
              <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${100 - (currentStack.latency / 500) * 100}%` }}
                  transition={{ duration: 0.8 }}
                  className={`h-full bg-gradient-to-r from-green-400 to-green-600`}
                />
              </div>
            </div>

            {/* 内存效率 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-medium">内存效率</span>
                <span className="font-mono font-bold text-slate-800">{currentStack.memoryEfficiency}%</span>
              </div>
              <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${currentStack.memoryEfficiency}%` }}
                  transition={{ duration: 0.8 }}
                  className={`h-full bg-gradient-to-r from-purple-400 to-purple-600`}
                />
              </div>
            </div>

            {/* 易用性 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-medium">易用性</span>
                <span className="font-mono font-bold text-slate-800">{currentStack.easeOfUse}%</span>
              </div>
              <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${currentStack.easeOfUse}%` }}
                  transition={{ duration: 0.8 }}
                  className={`h-full bg-gradient-to-r from-blue-400 to-blue-600`}
                />
              </div>
            </div>
          </div>
        </div>

        {/* 优缺点 */}
        <div className="grid grid-cols-2 gap-4">
          {/* 优点 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <h4 className="font-bold text-green-800">优点</h4>
            </div>
            <ul className="space-y-2">
              {currentStack.pros.map((pro, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-700"
                >
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>{pro}</span>
                </motion.li>
              ))}
            </ul>
          </div>

          {/* 缺点 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <div className="flex items-center gap-2 mb-3">
              <XCircle className="w-5 h-5 text-red-600" />
              <h4 className="font-bold text-red-800">缺点</h4>
            </div>
            <ul className="space-y-2">
              {currentStack.cons.map((con, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2 text-sm text-slate-700"
                >
                  <span className="text-red-600 mt-0.5">✗</span>
                  <span>{con}</span>
                </motion.li>
              ))}
            </ul>
          </div>
        </div>
      </motion.div>

      {/* 对比表格 */}
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">全面对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-slate-300">
                <th className="text-left py-3 px-4">特性</th>
                {stacks.map(stack => (
                  <th key={stack.id} className="text-center py-3 px-4">{stack.name.split('(')[0]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-200">
                <td className="py-3 px-4 font-medium">吞吐量</td>
                {stacks.map(stack => (
                  <td key={stack.id} className="text-center py-3 px-4">
                    <span className={`font-mono font-bold text-${stack.color}-600`}>
                      {stack.throughput}
                    </span>
                  </td>
                ))}
              </tr>
              <tr className="border-b border-slate-200">
                <td className="py-3 px-4 font-medium">延迟 (P95)</td>
                {stacks.map(stack => (
                  <td key={stack.id} className="text-center py-3 px-4">
                    <span className="font-mono">{stack.latency} ms</span>
                  </td>
                ))}
              </tr>
              <tr className="border-b border-slate-200">
                <td className="py-3 px-4 font-medium">内存效率</td>
                {stacks.map(stack => (
                  <td key={stack.id} className="text-center py-3 px-4">
                    <span className="font-mono">{stack.memoryEfficiency}%</span>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium">易用性</td>
                {stacks.map(stack => (
                  <td key={stack.id} className="text-center py-3 px-4">
                    <span className="font-mono">{stack.easeOfUse}%</span>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 选择建议 */}
      <div className="mt-6 bg-gradient-to-r from-indigo-50 to-purple-50 p-5 rounded-lg shadow border border-indigo-200">
        <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
          <Zap className="w-5 h-5 text-indigo-600" />
          选择建议
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="font-bold text-indigo-800 mb-2">场景推荐</div>
            <ul className="space-y-1 text-slate-700">
              <li>• <strong>原型开发</strong>: Transformers</li>
              <li>• <strong>生产高吞吐</strong>: vLLM</li>
              <li>• <strong>官方生态</strong>: TGI</li>
              <li>• <strong>极致性能</strong>: TensorRT-LLM</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-indigo-800 mb-2">性能优先级</div>
            <ul className="space-y-1 text-slate-700">
              <li>• 延迟敏感 → TensorRT-LLM</li>
              <li>• 吞吐量优先 → vLLM / TGI</li>
              <li>• 易用性 → Transformers / TGI</li>
              <li>• 显存受限 → vLLM + 量化</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

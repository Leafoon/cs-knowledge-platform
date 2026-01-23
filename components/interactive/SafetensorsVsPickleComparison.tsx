"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type ComparisonAspect = 'security' | 'speed' | 'compatibility' | 'features'

interface BenchmarkData {
  model: string
  pytorchTime: number
  safetensorsTime: number
  speedup: number
}

const SafetensorsVsPickleComparison: React.FC = () => {
  const [selectedAspect, setSelectedAspect] = useState<ComparisonAspect>('security')

  const benchmarks: BenchmarkData[] = [
    { model: 'BERT-base', pytorchTime: 3.2, safetensorsTime: 0.8, speedup: 4.0 },
    { model: 'GPT-2', pytorchTime: 2.1, safetensorsTime: 0.5, speedup: 4.2 },
    { model: 'LLaMA-7B', pytorchTime: 147, safetensorsTime: 32, speedup: 4.6 },
    { model: 'LLaMA-70B', pytorchTime: 1420, safetensorsTime: 285, speedup: 5.0 },
  ]

  const aspects = [
    { id: 'security' as ComparisonAspect, label: '安全性', icon: '🔒' },
    { id: 'speed' as ComparisonAspect, label: '加载速度', icon: '⚡' },
    { id: 'compatibility' as ComparisonAspect, label: '兼容性', icon: '🔄' },
    { id: 'features' as ComparisonAspect, label: '特性', icon: '✨' },
  ]

  const comparisonContent = {
    security: {
      pytorch: [
        { text: '使用 Python Pickle', risk: 'high' },
        { text: '可执行任意代码', risk: 'high' },
        { text: '易受恶意注入攻击', risk: 'high' },
        { text: '无法验证数据完整性', risk: 'medium' },
      ],
      safetensors: [
        { text: '纯数据格式（零代码执行）', risk: 'safe' },
        { text: '文件头包含完整元数据', risk: 'safe' },
        { text: '防止任意代码注入', risk: 'safe' },
        { text: '支持数据完整性校验', risk: 'safe' },
      ],
    },
    speed: {
      explanation: 'Safetensors 使用内存映射（mmap），支持零拷贝加载，速度提升 3-5x',
    },
    compatibility: {
      pytorch: [
        { text: 'PyTorch 版本敏感', status: 'warning' },
        { text: 'Python 版本依赖', status: 'warning' },
        { text: '跨平台兼容性差', status: 'warning' },
        { text: '不支持部分加载', status: 'error' },
      ],
      safetensors: [
        { text: '与 PyTorch 版本无关', status: 'success' },
        { text: '语言无关（纯数据）', status: 'success' },
        { text: '跨平台稳定', status: 'success' },
        { text: '支持按需加载张量', status: 'success' },
      ],
    },
    features: {
      pytorch: ['完整 Python 对象', '支持任意数据结构', '序列化复杂对象'],
      safetensors: [
        '仅存储张量数据',
        '支持元数据（JSON）',
        '内存映射（mmap）',
        '部分加载（lazy loading）',
        '零拷贝读取',
        '多框架支持（PyTorch/TensorFlow/JAX）',
      ],
    },
  }

  const maxTime = Math.max(...benchmarks.map((b) => b.pytorchTime))

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">Safetensors vs PyTorch Pickle</h3>
        <p className="text-gray-600 dark:text-gray-400">
          安全、快速的模型序列化格式对比
        </p>
      </div>

      {/* 方面选择 */}
      <div className="flex flex-wrap gap-2 justify-center">
        {aspects.map((aspect) => (
          <button
            key={aspect.id}
            onClick={() => setSelectedAspect(aspect.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selectedAspect === aspect.id
                ? 'bg-blue-500 text-white shadow-lg scale-105'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <span className="mr-2">{aspect.icon}</span>
            {aspect.label}
          </button>
        ))}
      </div>

      {/* 内容展示 */}
      <motion.div
        key={selectedAspect}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
      >
        {selectedAspect === 'security' && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* PyTorch Pickle */}
            <div>
              <h4 className="text-lg font-bold mb-4 flex items-center">
                <span className="text-2xl mr-2">⚠️</span>
                PyTorch Pickle (.bin)
              </h4>
              <div className="space-y-2">
                {comparisonContent.security.pytorch.map((item, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border-l-4 ${
                      item.risk === 'high'
                        ? 'bg-red-50 dark:bg-red-900/20 border-red-500'
                        : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500'
                    }`}
                  >
                    <p className="text-sm">{item.text}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-900 rounded-lg">
                <p className="text-xs font-mono text-gray-700 dark:text-gray-300">
                  # 恶意 Pickle 示例
                  <br />
                  <span className="text-red-600">import os</span>
                  <br />
                  <span className="text-red-600">
                    os.system(&apos;rm -rf /&apos;) # 危险！
                  </span>
                </p>
              </div>
            </div>

            {/* Safetensors */}
            <div>
              <h4 className="text-lg font-bold mb-4 flex items-center">
                <span className="text-2xl mr-2">✅</span>
                Safetensors
              </h4>
              <div className="space-y-2">
                {comparisonContent.security.safetensors.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg border-l-4 bg-green-50 dark:bg-green-900/20 border-green-500"
                  >
                    <p className="text-sm">{item.text}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-900 rounded-lg">
                <p className="text-xs font-mono text-gray-700 dark:text-gray-300">
                  # 纯数据格式
                  <br />
                  <span className="text-green-600">
                    [Header: metadata]
                  </span>
                  <br />
                  <span className="text-green-600">
                    [Tensors: binary data]
                  </span>
                </p>
              </div>
            </div>
          </div>
        )}

        {selectedAspect === 'speed' && (
          <div>
            <p className="text-center mb-6 text-gray-600 dark:text-gray-400">
              {comparisonContent.speed.explanation}
            </p>
            <div className="space-y-4">
              {benchmarks.map((benchmark, idx) => (
                <div key={idx} className="space-y-2">
                  <div className="flex items-center justify-between text-sm font-medium">
                    <span>{benchmark.model}</span>
                    <span className="text-green-600 dark:text-green-400">
                      {benchmark.speedup.toFixed(1)}x 加速
                    </span>
                  </div>
                  <div className="space-y-1">
                    {/* PyTorch */}
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs text-gray-600 dark:text-gray-400">
                        PyTorch
                      </span>
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(benchmark.pytorchTime / maxTime) * 100}%` }}
                          transition={{ duration: 1, delay: idx * 0.1 }}
                          className="h-full bg-red-500 flex items-center justify-end pr-2"
                        >
                          <span className="text-xs text-white font-medium">
                            {benchmark.pytorchTime}s
                          </span>
                        </motion.div>
                      </div>
                    </div>
                    {/* Safetensors */}
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs text-gray-600 dark:text-gray-400">
                        Safetensors
                      </span>
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(benchmark.safetensorsTime / maxTime) * 100}%` }}
                          transition={{ duration: 1, delay: idx * 0.1 }}
                          className="h-full bg-green-500 flex items-center justify-end pr-2"
                        >
                          <span className="text-xs text-white font-medium">
                            {benchmark.safetensorsTime}s
                          </span>
                        </motion.div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedAspect === 'compatibility' && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* PyTorch */}
            <div>
              <h4 className="text-lg font-bold mb-4">PyTorch Pickle</h4>
              <div className="space-y-2">
                {comparisonContent.compatibility.pytorch.map((item, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg flex items-start gap-2 ${
                      item.status === 'warning'
                        ? 'bg-yellow-50 dark:bg-yellow-900/20'
                        : 'bg-red-50 dark:bg-red-900/20'
                    }`}
                  >
                    <span className="text-xl">
                      {item.status === 'warning' ? '⚠️' : '❌'}
                    </span>
                    <p className="text-sm flex-1">{item.text}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Safetensors */}
            <div>
              <h4 className="text-lg font-bold mb-4">Safetensors</h4>
              <div className="space-y-2">
                {comparisonContent.compatibility.safetensors.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg flex items-start gap-2 bg-green-50 dark:bg-green-900/20"
                  >
                    <span className="text-xl">✅</span>
                    <p className="text-sm flex-1">{item.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedAspect === 'features' && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* PyTorch */}
            <div>
              <h4 className="text-lg font-bold mb-4">PyTorch Pickle</h4>
              <ul className="space-y-2">
                {comparisonContent.features.pytorch.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Safetensors */}
            <div>
              <h4 className="text-lg font-bold mb-4">Safetensors</h4>
              <ul className="space-y-2">
                {comparisonContent.features.safetensors.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">•</span>
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </motion.div>

      {/* 推荐卡片 */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6 border-2 border-green-500">
        <div className="flex items-start gap-4">
          <div className="text-4xl">💡</div>
          <div>
            <h4 className="font-bold text-lg mb-2">推荐使用 Safetensors</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              对于所有新项目和模型分享，<strong>始终优先使用 Safetensors</strong>。
              它提供了更高的安全性、更快的加载速度和更好的跨平台兼容性。
              Hugging Face Hub 已将 Safetensors 作为默认格式。
            </p>
            <div className="mt-3 flex gap-4 text-sm">
              <span className="text-green-600 dark:text-green-400">✅ 3-5x 加载加速</span>
              <span className="text-green-600 dark:text-green-400">✅ 零安全风险</span>
              <span className="text-green-600 dark:text-green-400">✅ 部分加载支持</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SafetensorsVsPickleComparison

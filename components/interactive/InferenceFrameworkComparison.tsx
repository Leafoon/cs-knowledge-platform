'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

export default function InferenceFrameworkComparison() {
  const [selectedMetric, setSelectedMetric] = useState<'throughput' | 'latency' | 'memory'>('throughput')

  const frameworks = [
    { name: 'Transformers', color: 'gray', description: '原生实现' },
    { name: 'TGI', color: 'blue', description: 'Text Generation Inference' },
    { name: 'vLLM', color: 'green', description: 'PagedAttention + Continuous Batching' },
  ]

  const benchmarkData = {
    throughput: {
      label: '吞吐量 (req/s)',
      unit: 'req/s',
      data: [
        { framework: 'Transformers', value: 2.3, color: 'gray' },
        { framework: 'TGI', value: 18.7, color: 'blue' },
        { framework: 'vLLM', value: 23.5, color: 'green' },
      ],
      max: 25,
      inverse: false,
    },
    latency: {
      label: 'P50 延迟 (秒)',
      unit: 's',
      data: [
        { framework: 'Transformers', value: 4.2, color: 'gray' },
        { framework: 'TGI', value: 0.9, color: 'blue' },
        { framework: 'vLLM', value: 0.7, color: 'green' },
      ],
      max: 5,
      inverse: true, // 越低越好
    },
    memory: {
      label: '显存占用 (GB)',
      unit: 'GB',
      data: [
        { framework: 'Transformers', value: 38.4, color: 'gray' },
        { framework: 'TGI', value: 22.3, color: 'blue' },
        { framework: 'vLLM', value: 19.1, color: 'green' },
      ],
      max: 40,
      inverse: true,
    },
  }

  const currentData = benchmarkData[selectedMetric]
  const bestValue = (currentData.inverse ?? false)
    ? Math.min(...currentData.data.map((d) => d.value))
    : Math.max(...currentData.data.map((d) => d.value))

  // 详细特性对比
  const featureComparison = [
    {
      feature: 'PagedAttention',
      transformers: '❌',
      tgi: '✅',
      vllm: '✅',
    },
    {
      feature: 'Continuous Batching',
      transformers: '❌',
      tgi: '✅',
      vllm: '✅',
    },
    {
      feature: 'Flash Attention 2',
      transformers: '⚠️ 需手动启用',
      tgi: '✅ 自动',
      vllm: '✅ 自动',
    },
    {
      feature: 'Tensor Parallelism',
      transformers: '❌',
      tgi: '✅',
      vllm: '✅',
    },
    {
      feature: '量化支持',
      transformers: 'bitsandbytes',
      tgi: 'bitsandbytes, GPTQ, AWQ',
      vllm: 'GPTQ, AWQ',
    },
    {
      feature: 'Streaming 生成',
      transformers: '✅',
      tgi: '✅ SSE',
      vllm: '✅ OpenAI API',
    },
    {
      feature: '部署复杂度',
      transformers: '简单',
      tgi: 'Docker',
      vllm: '中等',
    },
    {
      feature: '适用场景',
      transformers: '开发/调试',
      tgi: '生产部署',
      vllm: '高吞吐推理',
    },
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-900 dark:text-white">
        推理框架性能对比
      </h3>

      {/* 指标选择 */}
      <div className="flex gap-2 mb-6">
        {Object.entries(benchmarkData).map(([key, data]) => (
          <button
            key={key}
            onClick={() => setSelectedMetric(key as any)}
            className={`flex-1 px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedMetric === key
                ? 'bg-blue-500 text-white scale-105'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            {data.label}
          </button>
        ))}
      </div>

      {/* 性能图表 */}
      <div className="mb-8 p-6 bg-gray-50 dark:bg-gray-900 rounded-xl">
        <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          {currentData.label} {(currentData.inverse ?? false) && '（越低越好）'}
        </h4>

        <div className="space-y-4">
          {currentData.data.map((item, index) => {
            const percentage = (item.value / currentData.max) * 100
            const isBest = item.value === bestValue
            const speedup = (currentData.inverse ?? false)
              ? (currentData.data[0].value / item.value).toFixed(1)
              : (item.value / currentData.data[0].value).toFixed(1)

            return (
              <motion.div
                key={item.framework}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-gray-900 dark:text-white w-32">
                      {item.framework}
                    </span>
                    {isBest && (
                      <span className="text-xs px-2 py-1 bg-green-500 text-white rounded-full font-bold">
                        🏆 最佳
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`text-lg font-bold text-${item.color}-700 dark:text-${item.color}-300`}>
                      {item.value} {currentData.unit}
                    </span>
                    {index > 0 && (
                      <span className="text-sm text-gray-600 dark:text-gray-400 w-16 text-right">
                        {speedup}x
                      </span>
                    )}
                  </div>
                </div>

                <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full bg-${item.color}-500`}
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ delay: index * 0.1 + 0.2, duration: 0.5 }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* 详细特性对比表 */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-900">
              <th className="px-4 py-3 text-left font-bold text-gray-900 dark:text-white">
                特性
              </th>
              <th className="px-4 py-3 text-center font-bold text-gray-700 dark:text-gray-300">
                Transformers
              </th>
              <th className="px-4 py-3 text-center font-bold text-blue-700 dark:text-blue-300">
                TGI
              </th>
              <th className="px-4 py-3 text-center font-bold text-green-700 dark:text-green-300">
                vLLM
              </th>
            </tr>
          </thead>
          <tbody>
            {featureComparison.map((row, index) => (
              <tr
                key={index}
                className="border-b border-gray-200 dark:border-gray-700"
              >
                <td className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                  {row.feature}
                </td>
                <td className="px-4 py-3 text-center text-gray-700 dark:text-gray-300">
                  {row.transformers}
                </td>
                <td className="px-4 py-3 text-center text-blue-700 dark:text-blue-300">
                  {row.tgi}
                </td>
                <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                  {row.vllm}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 选择建议 */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-900/20 rounded-lg">
          <h5 className="font-semibold text-gray-900 dark:text-gray-300 mb-2">
            Transformers 原生
          </h5>
          <p className="text-sm text-gray-700 dark:text-gray-400 mb-2">
            适用场景：
          </p>
          <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
            <li>• 开发阶段快速迭代</li>
            <li>• 低并发（&lt; 5 请求）</li>
            <li>• 教学演示</li>
            <li>• 自定义模型</li>
          </ul>
        </div>

        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
            TGI
          </h5>
          <p className="text-sm text-blue-700 dark:text-blue-400 mb-2">
            适用场景：
          </p>
          <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
            <li>• 生产环境部署</li>
            <li>• Docker/K8s 集成</li>
            <li>• 官方支持优先</li>
            <li>• 企业级 SLA</li>
          </ul>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            vLLM
          </h5>
          <p className="text-sm text-green-700 dark:text-green-400 mb-2">
            适用场景：
          </p>
          <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
            <li>• 高吞吐量优先</li>
            <li>• 显存受限场景</li>
            <li>• 批量离线推理</li>
            <li>• Python 生态集成</li>
          </ul>
        </div>
      </div>

      {/* 测试环境说明 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">测试配置：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`模型：LLaMA-2-13B-Chat
硬件：8×A100 40GB
数据集：ShareGPT（2000 条对话）
平均生成长度：150 tokens
并发数：64`}
        </pre>
      </div>
    </div>
  )
}

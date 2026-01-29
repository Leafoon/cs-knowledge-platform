'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

type AttentionType = 'mha' | 'gqa' | 'mqa'

export default function MQAvsGQAComparison() {
  const [activeType, setActiveType] = useState<AttentionType>('mha')

  const numHeads = 32
  const headDim = 128
  const seqLen = 2048

  const configs = {
    mha: {
      name: 'Multi-Head Attention (MHA)',
      numKVHeads: 32,
      color: 'blue',
      description: '每个 head 独立的 K、V',
    },
    gqa: {
      name: 'Grouped-Query Attention (GQA)',
      numKVHeads: 8,
      color: 'purple',
      description: `${numHeads / 8} 个 Q heads 共享 1 组 K、V`,
    },
    mqa: {
      name: 'Multi-Query Attention (MQA)',
      numKVHeads: 1,
      color: 'green',
      description: '所有 heads 共享同一组 K、V',
    },
  }

  const currentConfig = configs[activeType]

  // 计算显存占用（假设 FP16）
  const calcMemory = (numKVHeads: number) => {
    // 2 (K+V) × layers × kv_heads × head_dim × seq_len × 2 bytes
    const layers = 32
    const bytesPerParam = 2
    return (2 * layers * numKVHeads * headDim * seqLen * bytesPerParam) / (1024 ** 3)
  }

  const mhaMemory = calcMemory(configs.mha.numKVHeads)
  const currentMemory = calcMemory(currentConfig.numKVHeads)
  const savings = ((mhaMemory - currentMemory) / mhaMemory * 100).toFixed(1)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        MHA vs GQA vs MQA 架构对比
      </h3>

      {/* 类型选择 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {(Object.keys(configs) as AttentionType[]).map((type) => (
          <button
            key={type}
            onClick={() => setActiveType(type)}
            className={`p-4 rounded-lg border-2 transition-all ${
              activeType === type
                ? `border-${configs[type].color}-500 bg-${configs[type].color}-50 dark:bg-${configs[type].color}-900/30 scale-105`
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
            }`}
          >
            <div className={`font-bold mb-1 ${
              activeType === type
                ? `text-${configs[type].color}-700 dark:text-${configs[type].color}-300`
                : 'text-gray-100'
            }`}>
              {configs[type].name.split(' (')[1]?.replace(')', '')}
            </div>
            <div className="text-xs text-gray-300 mb-2">
              KV Heads: {configs[type].numKVHeads}
            </div>
          </button>
        ))}
      </div>

      {/* 架构可视化 */}
      <div className={`mb-6 p-6 rounded-xl bg-${currentConfig.color}-50 dark:bg-${currentConfig.color}-900/20 border-2 border-${currentConfig.color}-300 dark:border-${currentConfig.color}-700`}>
        <h4 className={`text-lg font-bold text-${currentConfig.color}-900 dark:text-${currentConfig.color}-300 mb-3`}>
          {currentConfig.name}
        </h4>
        <p className={`text-sm text-${currentConfig.color}-700 dark:text-${currentConfig.color}-400 mb-4`}>
          {currentConfig.description}
        </p>

        <div className="space-y-4">
          {/* Q Heads */}
          <div>
            <p className="text-sm font-semibold text-gray-100 mb-2">
              Query Heads（{numHeads} 个）
            </p>
            <div className="grid grid-cols-16 gap-1">
              {Array.from({ length: numHeads }, (_, i) => (
                <div
                  key={`q-${i}`}
                  className={`h-8 rounded bg-${currentConfig.color}-400 flex items-center justify-center text-[10px] font-bold text-white`}
                >
                  Q{i}
                </div>
              ))}
            </div>
          </div>

          {/* KV Heads */}
          <div>
            <p className="text-sm font-semibold text-gray-100 mb-2">
              Key/Value Heads（{currentConfig.numKVHeads} 个）
            </p>
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${Math.min(currentConfig.numKVHeads, 16)}, minmax(0, 1fr))` }}>
              {Array.from({ length: currentConfig.numKVHeads }, (_, i) => (
                <motion.div
                  key={`kv-${i}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className={`h-16 rounded bg-${currentConfig.color}-600 flex flex-col items-center justify-center text-white`}
                >
                  <span className="text-xs font-bold">K{i}</span>
                  <span className="text-xs font-bold">V{i}</span>
                  {activeType === 'gqa' && (
                    <span className="text-[8px] mt-1">
                      (共享给 Q{i * 4}-Q{i * 4 + 3})
                    </span>
                  )}
                  {activeType === 'mqa' && (
                    <span className="text-[8px] mt-1">
                      (所有 Q 共享)
                    </span>
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* 连接示意 */}
          {activeType === 'gqa' && (
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg text-xs text-purple-800 dark:text-purple-200">
              💡 分组示例：Q0-Q3 共享 K0/V0，Q4-Q7 共享 K1/V1，...
            </div>
          )}
          {activeType === 'mqa' && (
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg text-xs text-green-800 dark:text-green-200">
              💡 所有 32 个 Q heads 都使用同一组 K0/V0
            </div>
          )}
        </div>
      </div>

      {/* 性能对比表 */}
      <div className="overflow-x-auto mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-900">
              <th className="px-4 py-3 text-left font-bold text-gray-100">
                指标
              </th>
              <th className="px-4 py-3 text-center font-bold text-blue-700 dark:text-blue-300">
                MHA
              </th>
              <th className="px-4 py-3 text-center font-bold text-purple-700 dark:text-purple-300">
                GQA (8 groups)
              </th>
              <th className="px-4 py-3 text-center font-bold text-green-700 dark:text-green-300">
                MQA
              </th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-4 py-3 font-semibold text-gray-100">
                KV Heads
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                32
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                8
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                1
              </td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-4 py-3 font-semibold text-gray-100">
                KV Cache 显存（2048 tokens）
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                {calcMemory(32).toFixed(2)} GB
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                {calcMemory(8).toFixed(2)} GB
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                {calcMemory(1).toFixed(3)} GB
              </td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-4 py-3 font-semibold text-gray-100">
                显存节省
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                0%
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                75%
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                96.8%
              </td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-4 py-3 font-semibold text-gray-100">
                推理速度提升
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                1.0x
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                1.14x
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                1.21x
              </td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-4 py-3 font-semibold text-gray-100">
                Perplexity 变化
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                5.68
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                5.72 (+0.7%)
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                5.89 (+3.7%)
              </td>
            </tr>
            <tr>
              <td className="px-4 py-3 font-semibold text-gray-100">
                是否需要重新训练
              </td>
              <td className="px-4 py-3 text-center text-gray-100">
                -
              </td>
              <td className="px-4 py-3 text-center text-purple-700 dark:text-purple-300">
                ✅ 需要
              </td>
              <td className="px-4 py-3 text-center text-green-700 dark:text-green-300">
                ✅ 需要
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 显存对比图 */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-xl">
        <h5 className="text-lg font-bold text-gray-100 mb-4">
          KV Cache 显存占用对比（LLaMA-7B，2048 tokens）
        </h5>

        <div className="space-y-3">
          {(['mha', 'gqa', 'mqa'] as AttentionType[]).map((type, index) => {
            const config = configs[type]
            const memory = calcMemory(config.numKVHeads)
            const percentage = (memory / mhaMemory) * 100

            return (
              <div key={type} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-gray-100">
                    {config.name.split(' (')[1]?.replace(')', '')}
                  </span>
                  <span className={`text-${config.color}-700 dark:text-${config.color}-300 font-bold`}>
                    {memory.toFixed(3)} GB
                  </span>
                </div>
                <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ delay: index * 0.1, duration: 0.5 }}
                    className={`h-full bg-${config.color}-500`}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 使用模型 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
            使用 MHA 的模型
          </h5>
          <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
            <li>• LLaMA-1（所有版本）</li>
            <li>• GPT-3</li>
            <li>• BERT</li>
            <li>• T5</li>
          </ul>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <h5 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
            使用 GQA 的模型
          </h5>
          <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
            <li>• LLaMA-2-70B ⭐</li>
            <li>• Mistral-7B ⭐</li>
            <li>• Qwen-7B</li>
            <li>• CodeLLaMA-34B</li>
          </ul>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">
            使用 MQA 的模型
          </h5>
          <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
            <li>• PaLM（Google）</li>
            <li>• Falcon-40B</li>
            <li>• StarCoder</li>
            <li>• Chinchilla</li>
          </ul>
        </div>
      </div>

      {/* 公式说明 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">KV Cache 显存计算：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`Memory_KV = 2 × n_layers × num_kv_heads × head_dim × seq_len × 2 bytes

MHA:  2 × 32 × 32 × 128 × 2048 × 2 = 1.07 GB
GQA:  2 × 32 × 8  × 128 × 2048 × 2 = 0.27 GB（节省 75%）
MQA:  2 × 32 × 1  × 128 × 2048 × 2 = 0.03 GB（节省 96.8%）`}
        </pre>
      </div>
    </div>
  )
}

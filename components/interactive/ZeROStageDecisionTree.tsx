'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function ZeROStageDecisionTree() {
  const [modelSize, setModelSize] = useState<string>('')
  const [gpuMemory, setGpuMemory] = useState<string>('')
  const [numGPUs, setNumGPUs] = useState<number>(1)

  const getRecommendation = () => {
    if (!modelSize || !gpuMemory) return null

    const model = parseFloat(modelSize)
    const memory = parseFloat(gpuMemory)
    const totalMemory = memory * numGPUs

    // 决策逻辑
    if (model < 3) {
      return {
        stage: 'ZeRO-1 或标准 DDP',
        reason: '模型较小，单卡或少量 GPU 即可',
        color: 'green',
        offload: '不需要',
        expectedMemory: `${(model * 2 * 1.5).toFixed(1)} GB/GPU`
      }
    } else if (model >= 3 && model < 13) {
      if (totalMemory >= model * 2 * 1.5 * 4) {
        return {
          stage: 'ZeRO-2',
          reason: '显存充足，ZeRO-2 提供最佳性能',
          color: 'blue',
          offload: '不需要',
          expectedMemory: `${(model * 2 * 1.5 / numGPUs * 1.2).toFixed(1)} GB/GPU`
        }
      } else {
        return {
          stage: 'ZeRO-3',
          reason: '显存紧张，需要完全分片',
          color: 'purple',
          offload: '可选 CPU Offload',
          expectedMemory: `${(model * 2 * 1.5 / numGPUs).toFixed(1)} GB/GPU`
        }
      }
    } else if (model >= 13 && model < 70) {
      if (memory >= 80) {
        return {
          stage: 'ZeRO-3',
          reason: '大显存 GPU，ZeRO-3 可高效训练',
          color: 'purple',
          offload: '不需要',
          expectedMemory: `${(model * 2 * 1.5 / numGPUs).toFixed(1)} GB/GPU`
        }
      } else {
        return {
          stage: 'ZeRO-3 + CPU Offload',
          reason: '显存不足，需要 Offload 优化器状态',
          color: 'orange',
          offload: 'CPU Offload 必需',
          expectedMemory: `${(model * 2 / numGPUs).toFixed(1)} GB/GPU`
        }
      }
    } else {
      if (memory >= 80 && numGPUs >= 8) {
        return {
          stage: 'ZeRO-3 + CPU Offload',
          reason: '超大模型，建议 Offload',
          color: 'orange',
          offload: 'CPU Offload 推荐',
          expectedMemory: `${(model * 2 / numGPUs * 0.7).toFixed(1)} GB/GPU`
        }
      } else {
        return {
          stage: 'ZeRO-3 + CPU/NVMe Offload',
          reason: '显存极度受限，需要 NVMe',
          color: 'red',
          offload: 'NVMe Offload 必需',
          expectedMemory: `${(model * 2 / numGPUs * 0.3).toFixed(1)} GB/GPU`
        }
      }
    }
  }

  const recommendation = getRecommendation()

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-900 dark:text-white">
        ZeRO Stage 决策树
      </h3>

      {/* 输入参数 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            模型参数量（B）
          </label>
          <input
            type="number"
            value={modelSize}
            onChange={(e) => setModelSize(e.target.value)}
            placeholder="例如：7"
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            GPT-2 XL: 1.3, LLaMA-7B: 7, LLaMA-13B: 13
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            单 GPU 显存（GB）
          </label>
          <select
            value={gpuMemory}
            onChange={(e) => setGpuMemory(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="">选择显存</option>
            <option value="16">16 GB (V100)</option>
            <option value="24">24 GB (RTX 3090)</option>
            <option value="40">40 GB (A100)</option>
            <option value="80">80 GB (A100)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GPU 数量
          </label>
          <input
            type="number"
            value={numGPUs}
            onChange={(e) => setNumGPUs(Math.max(1, parseInt(e.target.value) || 1))}
            min="1"
            max="64"
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>
      </div>

      {/* 推荐结果 */}
      <AnimatePresence mode="wait">
        {recommendation && (
          <motion.div
            key={recommendation.stage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className={`p-6 rounded-xl border-2 border-${recommendation.color}-500 bg-${recommendation.color}-50 dark:bg-${recommendation.color}-900/20`}
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  推荐配置：{recommendation.stage}
                </h4>
                <p className="text-gray-700 dark:text-gray-300">
                  {recommendation.reason}
                </p>
              </div>
              <div className={`px-4 py-2 rounded-full bg-${recommendation.color}-500 text-white font-bold`}>
                {recommendation.color === 'green' && '✓ 简单'}
                {recommendation.color === 'blue' && '⚡ 推荐'}
                {recommendation.color === 'purple' && '🔧 高级'}
                {recommendation.color === 'orange' && '⚠️ 复杂'}
                {recommendation.color === 'red' && '🔴 极限'}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Offload 策略</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {recommendation.offload}
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">预计显存占用</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {recommendation.expectedMemory}
                </p>
              </div>
            </div>

            {/* 配置示例 */}
            <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
              <p className="text-xs text-gray-400 mb-2">ds_config.json 配置片段：</p>
              <pre className="text-sm text-green-400 overflow-x-auto">
{`{
  "zero_optimization": {
    "stage": ${recommendation.stage.includes('ZeRO-3') ? '3' : recommendation.stage.includes('ZeRO-2') ? '2' : '1'},${recommendation.offload.includes('CPU') ? `
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },${recommendation.offload.includes('NVMe') ? `
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    },` : ''}` : ''}
    "overlap_comm": true
  }
}`}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 决策树可视化 */}
      {!recommendation && (
        <div className="text-center text-gray-500 dark:text-gray-400 py-12">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p>请输入模型参数量和 GPU 配置以获取推荐</p>
        </div>
      )}

      {/* 说明 */}
      <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">💡 决策依据</h5>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
          <li>• <strong>&lt; 3B</strong>: ZeRO-1 或标准 DDP 即可</li>
          <li>• <strong>3B-13B</strong>: ZeRO-2 平衡性能与显存</li>
          <li>• <strong>13B-70B</strong>: ZeRO-3 必需，根据显存决定是否 Offload</li>
          <li>• <strong>&gt; 70B</strong>: ZeRO-3 + CPU/NVMe Offload</li>
        </ul>
      </div>
    </div>
  )
}

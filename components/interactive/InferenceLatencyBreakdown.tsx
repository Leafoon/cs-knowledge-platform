'use client'

import { motion } from 'framer-motion'

export default function InferenceLatencyBreakdown() {
  const stages = [
    { name: 'Tokenization', percent: 5, color: 'gray', time: '12ms', optimization: 'Fast Tokenizer (Rust)' },
    { name: 'Embedding', percent: 5, color: 'blue', time: '12ms', optimization: '无需优化' },
    { name: 'Attention', percent: 65, color: 'red', time: '156ms', optimization: 'Flash Attention 2' },
    { name: 'FFN', percent: 20, color: 'orange', time: '48ms', optimization: 'Kernel Fusion, torch.compile' },
    { name: 'Sampling', percent: 5, color: 'purple', time: '12ms', optimization: 'Static KV Cache' },
  ]

  const totalTime = stages.reduce((sum, stage) => sum + parseInt(stage.time), 0)

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        Transformer 推理延迟分解
      </h3>

      <div className="mb-8">
        <div className="text-center mb-4">
          <span className="text-3xl font-bold text-gray-100">{totalTime} ms</span>
          <p className="text-sm text-gray-300">总推理延迟（单 token）</p>
        </div>

        {/* 可视化进度条 */}
        <div className="flex h-16 rounded-lg overflow-hidden shadow-lg">
          {stages.map((stage, index) => (
            <motion.div
              key={stage.name}
              initial={{ width: 0 }}
              animate={{ width: `${stage.percent}%` }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              className={`bg-${stage.color}-500 flex items-center justify-center text-white text-sm font-semibold relative group cursor-pointer`}
              style={{ width: `${stage.percent}%` }}
            >
              <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-20 transition-opacity"></div>
              <span className={stage.percent < 10 ? 'text-xs' : ''}>
                {stage.percent}%
              </span>

              {/* Tooltip */}
              <div className="absolute bottom-full mb-2 hidden group-hover:block bg-gray-900 text-white text-xs rounded py-1 px-2 whitespace-nowrap z-10">
                {stage.name}: {stage.time}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 详细分解表 */}
      <div className="space-y-3">
        {stages.map((stage, index) => (
          <motion.div
            key={stage.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`p-4 rounded-lg border-2 border-${stage.color}-300 dark:border-${stage.color}-700 bg-${stage.color}-50 dark:bg-${stage.color}-900/20`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded-full bg-${stage.color}-500`}></div>
                <span className="font-bold text-gray-100">{stage.name}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className={`text-lg font-semibold text-${stage.color}-700 dark:text-${stage.color}-300`}>
                  {stage.time}
                </span>
                <span className={`px-3 py-1 rounded-full bg-${stage.color}-500 text-white text-sm`}>
                  {stage.percent}%
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className={`text-${stage.color}-700 dark:text-${stage.color}-300 font-semibold`}>
                优化方法:
              </span>
              <span className={`text-${stage.color}-600 dark:text-${stage.color}-400`}>
                {stage.optimization}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* 优化建议 */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">🎯 优化重点</h5>
          <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
            <li>• <strong>Attention (65%)</strong>: Flash Attention 可减少 40%-60%</li>
            <li>• <strong>FFN (20%)</strong>: torch.compile 可减少 30%-50%</li>
            <li>• 优化这两项可获得 <strong>2-3x</strong> 加速</li>
          </ul>
        </div>

        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">📊 实测数据</h5>
          <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
            <li>• 模型：LLaMA-7B</li>
            <li>• 硬件：A100 GPU</li>
            <li>• 序列长度：512 tokens</li>
            <li>• 精度：FP16</li>
          </ul>
        </div>
      </div>

      {/* 优化后效果对比 */}
      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
        <h5 className="font-semibold text-purple-900 dark:text-purple-300 mb-3">🚀 优化后性能提升</h5>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">原始</p>
            <p className="text-2xl font-bold text-purple-900 dark:text-purple-200">{totalTime}ms</p>
          </div>
          <div>
            <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">优化后</p>
            <p className="text-2xl font-bold text-green-600 dark:text-green-400">56ms</p>
          </div>
          <div>
            <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">加速比</p>
            <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">4.3x</p>
          </div>
        </div>
        <p className="text-xs text-center text-purple-600 dark:text-purple-400 mt-3">
          Flash Attention 2 + torch.compile + Static Cache
        </p>
      </div>
    </div>
  )
}

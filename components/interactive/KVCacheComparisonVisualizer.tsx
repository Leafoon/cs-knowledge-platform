'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

export default function KVCacheComparisonVisualizer() {
  const [generatedTokens, setGeneratedTokens] = useState(3)

  const maxTokens = 10
  const tokens = Array.from({ length: generatedTokens }, (_, i) => `T${i + 1}`)

  // 动态 Cache 显存占用（每次扩展）
  const dynamicMemory = tokens.map((_, i) => (i + 1) * 0.5) // MB
  const totalDynamicMemory = dynamicMemory.reduce((a, b) => a + b, 0)

  // 静态 Cache 显存占用（固定）
  const staticMemory = maxTokens * 0.5 // MB

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-900 dark:text-white">
        KV Cache 对比：动态 vs 静态
      </h3>

      {/* 控制滑块 */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          已生成 Token 数量: {generatedTokens}
        </label>
        <input
          type="range"
          min="1"
          max={maxTokens}
          value={generatedTokens}
          onChange={(e) => setGeneratedTokens(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* 可视化对比 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* 动态 Cache */}
        <div className="p-6 bg-orange-50 dark:bg-orange-900/20 rounded-xl border-2 border-orange-300 dark:border-orange-700">
          <h4 className="text-lg font-bold text-orange-900 dark:text-orange-300 mb-4">
            ⚠️ 动态 KV Cache（默认）
          </h4>

          <div className="space-y-2 mb-4">
            {tokens.map((token, i) => (
              <motion.div
                key={token}
                initial={{ opacity: 0, x: -20, height: 0 }}
                animate={{ opacity: 1, x: 0, height: 'auto' }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-3"
              >
                <div className="flex-1 bg-orange-500 text-white rounded-lg p-2 text-center text-sm font-semibold">
                  {token}
                </div>
                <div className="text-xs text-orange-700 dark:text-orange-400">
                  +{dynamicMemory[i].toFixed(1)} MB
                </div>
              </motion.div>
            ))}
          </div>

          <div className="p-3 bg-orange-600 text-white rounded-lg text-center">
            <p className="text-sm">总显存占用</p>
            <p className="text-2xl font-bold">{totalDynamicMemory.toFixed(1)} MB</p>
            <p className="text-xs mt-1">每次 cat() 分配新内存</p>
          </div>

          <div className="mt-4 space-y-2 text-xs text-orange-800 dark:text-orange-200">
            <div className="flex items-center gap-2">
              <span className="text-red-600">❌</span>
              <span>内存碎片化</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-red-600">❌</span>
              <span>动态 shape（无法优化）</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-red-600">❌</span>
              <span>频繁内存分配开销</span>
            </div>
          </div>
        </div>

        {/* 静态 Cache */}
        <div className="p-6 bg-green-50 dark:bg-green-900/20 rounded-xl border-2 border-green-300 dark:border-green-700">
          <h4 className="text-lg font-bold text-green-900 dark:text-green-300 mb-4">
            ✅ 静态 KV Cache（优化）
          </h4>

          <div className="mb-4">
            <div className="bg-green-200 dark:bg-green-800 rounded-lg p-3 mb-2">
              <p className="text-xs text-green-800 dark:text-green-200 mb-2 text-center">
                预分配固定大小：{maxTokens} tokens
              </p>
              <div className="grid grid-cols-10 gap-1">
                {Array.from({ length: maxTokens }, (_, i) => (
                  <div
                    key={i}
                    className={`h-8 rounded ${
                      i < generatedTokens
                        ? 'bg-green-500 text-white flex items-center justify-center text-[10px] font-bold'
                        : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  >
                    {i < generatedTokens && `T${i + 1}`}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-between text-xs text-green-700 dark:text-green-400">
              <span>已使用: {generatedTokens}/{maxTokens}</span>
              <span>利用率: {(generatedTokens / maxTokens * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="p-3 bg-green-600 text-white rounded-lg text-center">
            <p className="text-sm">总显存占用</p>
            <p className="text-2xl font-bold">{staticMemory.toFixed(1)} MB</p>
            <p className="text-xs mt-1">固定分配，零扩展开销</p>
          </div>

          <div className="mt-4 space-y-2 text-xs text-green-800 dark:text-green-200">
            <div className="flex items-center gap-2">
              <span className="text-green-600">✓</span>
              <span>零内存分配开销</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-green-600">✓</span>
              <span>固定 shape（GPU 优化）</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-green-600">✓</span>
              <span>与 torch.compile 完美配合</span>
            </div>
          </div>
        </div>
      </div>

      {/* 性能对比 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-sm text-blue-700 dark:text-blue-400 mb-1">显存节省</p>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-200">
            {((staticMemory - totalDynamicMemory) / totalDynamicMemory * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
            {staticMemory < totalDynamicMemory ? '节省' : '额外占用'}
          </p>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">速度提升</p>
          <p className="text-2xl font-bold text-purple-900 dark:text-purple-200">
            1.2-1.5x
          </p>
          <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
            减少内存分配开销
          </p>
        </div>

        <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg text-center">
          <p className="text-sm text-orange-700 dark:text-orange-400 mb-1">与 compile 组合</p>
          <p className="text-2xl font-bold text-orange-900 dark:text-orange-200">
            1.5x+
          </p>
          <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
            固定 shape 加速
          </p>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">启用静态 KV Cache：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`from transformers import StaticCache

# 创建静态 cache
cache = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=512,
    device="cuda",
    dtype=torch.float16
)

# 推理时传入
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    past_key_values=cache
)`}
        </pre>
      </div>

      {/* 最佳实践 */}
      <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
        <h5 className="font-semibold text-green-900 dark:text-green-300 mb-2">💡 何时使用静态 Cache？</h5>
        <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
          <li>• <strong>固定生成长度</strong>（如摘要、翻译任务）</li>
          <li>• <strong>结合 torch.compile</strong>（固定 shape 优化）</li>
          <li>• <strong>批量推理</strong>（减少内存碎片）</li>
          <li>• <strong>生产环境</strong>（稳定性能优先）</li>
        </ul>
      </div>
    </div>
  )
}

'use client'

import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface CacheState {
  layer: number
  seqLen: number
  kvSize: number
}

export default function KVCacheDynamics() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [tokens, setTokens] = useState<string[]>(['[CLS]', 'Hello'])
  const [cacheStates, setCacheStates] = useState<CacheState[]>([])
  const [batchSize, setBatchSize] = useState(1)
  const [nLayers, setNLayers] = useState(12)
  const [dModel, setDModel] = useState(768)
  const [showMemory, setShowMemory] = useState(true)

  const maxNewTokens = 10
  const exampleTokens = useMemo(() => ['world', '!', 'How', 'are', 'you', '?', 'I', 'am', 'fine', '.'], [])

  // 计算内存占用 (MB)
  const calculateMemory = useCallback((seqLen: number) => {
    // KV Cache: batch_size × n_layers × 2 (K+V) × seq_len × d_model × 2 bytes (fp16)
    const bytes = batchSize * nLayers * 2 * seqLen * dModel * 2
    return bytes / (1024 * 1024)
  }, [batchSize, nLayers, dModel])

  // 生成过程
  useEffect(() => {
    if (!isGenerating) return

    if (currentStep >= maxNewTokens) {
      setIsGenerating(false)
      return
    }

    const timer = setTimeout(() => {
      const newToken = exampleTokens[currentStep]
      setTokens(prev => [...prev, newToken])
      setCurrentStep(prev => prev + 1)

      // 更新 Cache 状态
      const newSeqLen = tokens.length + 1
      const newStates: CacheState[] = Array(nLayers).fill(null).map((_, i) => ({
        layer: i,
        seqLen: newSeqLen,
        kvSize: calculateMemory(newSeqLen) / nLayers,
      }))
      setCacheStates(newStates)
    }, 800)

    return () => clearTimeout(timer)
  }, [isGenerating, currentStep, tokens.length, calculateMemory, exampleTokens, nLayers])

  const reset = () => {
    setIsGenerating(false)
    setCurrentStep(0)
    setTokens(['[CLS]', 'Hello'])
    setCacheStates([])
  }

  const totalMemory = calculateMemory(tokens.length)
  const memoryPerToken = tokens.length > 0 ? totalMemory / tokens.length : 0

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          💾 KV Cache 动态增长可视化
        </h3>
        <p className="text-slate-600">
          逐 Token 观察 KV Cache 的动态扩展过程
        </p>
      </div>

      {/* 控制面板 */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            层数：{nLayers}
          </label>
          <input
            type="range"
            min="6"
            max="24"
            step="6"
            value={nLayers}
            onChange={(e) => setNLayers(Number(e.target.value))}
            className="w-full"
            disabled={isGenerating}
          />
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            模型维度：{dModel}
          </label>
          <input
            type="range"
            min="512"
            max="2048"
            step="256"
            value={dModel}
            onChange={(e) => setDModel(Number(e.target.value))}
            className="w-full"
            disabled={isGenerating}
          />
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Batch Size：{batchSize}
          </label>
          <input
            type="range"
            min="1"
            max="8"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            className="w-full"
            disabled={isGenerating}
          />
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6 justify-center">
        <button
          onClick={() => setIsGenerating(!isGenerating)}
          disabled={currentStep >= maxNewTokens}
          className={`px-6 py-2 rounded-lg font-medium text-white ${
            isGenerating
              ? 'bg-red-600 hover:bg-red-700'
              : 'bg-green-600 hover:bg-green-700 disabled:bg-slate-400'
          }`}
        >
          {isGenerating ? '⏸️ 暂停生成' : '▶️ 开始生成'}
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 rounded-lg font-medium bg-slate-600 text-white hover:bg-slate-700"
        >
          🔄 重置
        </button>
        <button
          onClick={() => setShowMemory(!showMemory)}
          className="px-6 py-2 rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700"
        >
          {showMemory ? '📊 隐藏内存' : '📊 显示内存'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：Token 序列 */}
        <div className="bg-white rounded-lg border border-slate-200 p-5">
          <h4 className="text-lg font-semibold text-slate-800 mb-4">
            🔤 生成的 Token 序列
          </h4>
          
          <div className="space-y-2 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {tokens.map((token, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`p-3 rounded-lg border ${
                    i < 2
                      ? 'bg-blue-50 border-blue-200'
                      : 'bg-green-50 border-green-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm font-medium text-slate-700">
                        {i < 2 ? '📝 Prompt' : '✨ Generated'}
                      </span>
                      <div className="text-lg font-bold text-slate-800 mt-1">
                        {token}
                      </div>
                    </div>
                    <div className="text-2xl font-bold text-slate-400">
                      {i}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          <div className="mt-4 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className="text-sm font-medium text-slate-700">
              当前序列长度
            </div>
            <div className="text-3xl font-bold text-blue-600 mt-1">
              {tokens.length}
            </div>
          </div>
        </div>

        {/* 中间：KV Cache 可视化 */}
        <div className="bg-white rounded-lg border border-slate-200 p-5">
          <h4 className="text-lg font-semibold text-slate-800 mb-4">
            💾 KV Cache 增长（第 1 层）
          </h4>
          
          <div className="space-y-3">
            {/* K Cache */}
            <div>
              <div className="text-sm font-medium text-slate-700 mb-2">
                Key Cache
              </div>
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: tokens.length }, (_, i) => (
                  <motion.div
                    key={`k-${i}`}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                    className="w-8 h-8 bg-blue-500 rounded flex items-center justify-center text-white text-xs font-bold"
                  >
                    K{i}
                  </motion.div>
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                形状: [{batchSize}, {tokens.length}, {dModel}]
              </div>
            </div>

            {/* V Cache */}
            <div>
              <div className="text-sm font-medium text-slate-700 mb-2">
                Value Cache
              </div>
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: tokens.length }, (_, i) => (
                  <motion.div
                    key={`v-${i}`}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                    className="w-8 h-8 bg-green-500 rounded flex items-center justify-center text-white text-xs font-bold"
                  >
                    V{i}
                  </motion.div>
                ))}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                形状: [{batchSize}, {tokens.length}, {dModel}]
              </div>
            </div>

            {/* 增长动画 */}
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg border border-blue-200">
              <div className="text-sm font-medium text-slate-700 mb-2">
                Cache 更新方式
              </div>
              <div className="text-xs text-slate-600 space-y-1">
                <div>• <strong>第 1 次</strong>：计算 Prompt 的 K, V</div>
                <div>• <strong>第 2+ 次</strong>：只计算新 token 的 K, V</div>
                <div>• <strong>拼接</strong>：torch.cat([past_kv, new_kv], dim=1)</div>
              </div>
            </div>
          </div>
        </div>

        {/* 右侧：统计信息 */}
        <div className="space-y-4">
          {/* 内存占用 */}
          {showMemory && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg border border-slate-200 p-5"
            >
              <h4 className="text-lg font-semibold text-slate-800 mb-4">
                📊 内存占用
              </h4>
              
              <div className="space-y-3">
                <div>
                  <div className="text-sm text-slate-600 mb-1">
                    总内存
                  </div>
                  <div className="text-3xl font-bold text-red-600">
                    {totalMemory.toFixed(1)} MB
                  </div>
                </div>

                <div>
                  <div className="text-sm text-slate-600 mb-1">
                    每个 Token
                  </div>
                  <div className="text-2xl font-bold text-orange-600">
                    {memoryPerToken.toFixed(2)} MB
                  </div>
                </div>

                <div>
                  <div className="text-sm text-slate-600 mb-1">
                    内存增长曲线
                  </div>
                  <svg width="100%" height="100" className="bg-slate-50 rounded">
                    <polyline
                      points={Array.from({ length: tokens.length }, (_, i) => {
                        const x = (i / Math.max(tokens.length - 1, 1)) * 250
                        const y = 90 - (calculateMemory(i + 1) / totalMemory) * 80
                        return `${x},${y}`
                      }).join(' ')}
                      fill="none"
                      stroke="#dc2626"
                      strokeWidth="3"
                    />
                  </svg>
                </div>
              </div>
            </motion.div>
          )}

          {/* 多层 Cache */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              🏢 多层 KV Cache
            </h4>
            
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {Array.from({ length: nLayers }, (_, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 p-2 bg-slate-50 rounded border border-slate-200"
                >
                  <div className="w-20 text-sm font-medium text-slate-700">
                    Layer {i}
                  </div>
                  <div className="flex-1">
                    <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-blue-500 to-green-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(tokens.length / (tokens.length + maxNewTokens - currentStep)) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="w-16 text-xs text-slate-600 text-right">
                    {tokens.length}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="text-xs text-purple-800">
                <strong>total_kv_cache</strong> = tuple of {nLayers} layers<br />
                每层: (K, V) × [{batchSize}, {tokens.length}, {dModel}]
              </div>
            </div>
          </div>

          {/* 性能对比 */}
          <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border border-green-200 p-5">
            <h4 className="text-lg font-semibold text-green-900 mb-3">
              ⚡ 性能对比
            </h4>
            
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-green-200">
                  <th className="text-left p-2">方法</th>
                  <th className="text-right p-2">速度</th>
                  <th className="text-right p-2">内存</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-green-100">
                  <td className="p-2">无 Cache</td>
                  <td className="text-right p-2 text-red-600">O(n²)</td>
                  <td className="text-right p-2 text-green-600">O(1)</td>
                </tr>
                <tr>
                  <td className="p-2">有 Cache</td>
                  <td className="text-right p-2 text-green-600">O(n)</td>
                  <td className="text-right p-2 text-red-600">O(n)</td>
                </tr>
              </tbody>
            </table>

            <div className="mt-3 text-xs text-green-800">
              对于长序列（n&gt;50），KV Cache 加速 <strong>10-100x</strong>
            </div>
          </div>

          {/* 代码示例 */}
          <div className="bg-slate-900 rounded-lg p-4 text-white">
            <div className="text-sm font-semibold mb-2">
              💻 使用示例
            </div>
            <pre className="text-xs overflow-x-auto">
              <code className="text-green-400">{`# 生成时启用 KV Cache
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True  # 关键！
)

# 手动管理 Cache
past_kv = None
for _ in range(max_new_tokens):
    outputs = model(
        input_ids[:, -1:],
        past_key_values=past_kv,
        use_cache=True
    )
    past_kv = outputs.past_key_values
    # past_kv 自动增长
`}</code>
            </pre>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm font-medium text-blue-900 mb-2">
          💡 KV Cache 核心要点
        </div>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>作用</strong>：缓存已计算的 K 和 V，避免重复计算</li>
          <li>• <strong>结构</strong>：past_key_values = ((K0, V0), (K1, V1), ...) 每层一对</li>
          <li>• <strong>更新</strong>：每次只计算新 token，然后 cat 到历史 cache</li>
          <li>• <strong>权衡</strong>：用内存换速度（长序列加速显著）</li>
          <li>• <strong>优化</strong>：PagedAttention（vLLM）动态分页管理，节省 50% 内存</li>
        </ul>
      </div>
    </div>
  )
}

'use client'

import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

type Phase = 'draft' | 'verify' | 'accept' | 'reject'

interface Token {
  id: number
  text: string
  draftProb: number
  targetProb: number
  status: 'pending' | 'accepted' | 'rejected' | 'verifying'
}

export default function SpeculativeDecodingFlowVisualizer() {
  const [currentPhase, setCurrentPhase] = useState<Phase>('draft')
  const [isPlaying, setIsPlaying] = useState(false)
  const [tokens, setTokens] = useState<Token[]>([])
  const [acceptedCount, setAcceptedCount] = useState(0)
  const [K, setK] = useState(5)

  const sampleTokens = useMemo(() => [
    { text: 'Once', draftProb: 0.85, targetProb: 0.92 },
    { text: 'upon', draftProb: 0.78, targetProb: 0.81 },
    { text: 'a', draftProb: 0.92, targetProb: 0.95 },
    { text: 'time', draftProb: 0.65, targetProb: 0.72 },
    { text: 'there', draftProb: 0.42, targetProb: 0.38 },
    { text: 'lived', draftProb: 0.58, targetProb: 0.61 },
    { text: 'was', draftProb: 0.71, targetProb: 0.68 },
  ], [])

  const reset = () => {
    setIsPlaying(false)
    setCurrentPhase('draft')
    setTokens([])
    setAcceptedCount(0)
  }

  useEffect(() => {
    if (!isPlaying) return

    const runDemo = async () => {
      // Phase 1: Draft Model 生成
      setCurrentPhase('draft')
      const draftTokens = sampleTokens.slice(0, K).map((t, i) => ({
        id: i,
        ...t,
        status: 'pending' as const,
      }))
      
      for (let i = 0; i < draftTokens.length; i++) {
        await new Promise((resolve) => setTimeout(resolve, 400))
        setTokens((prev) => [...prev, draftTokens[i]])
      }

      await new Promise((resolve) => setTimeout(resolve, 800))

      // Phase 2: Verify 验证
      setCurrentPhase('verify')
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Phase 3: Accept/Reject
      let accepted = 0
      for (let i = 0; i < draftTokens.length; i++) {
        const token = draftTokens[i]
        const acceptProb = Math.min(1, token.targetProb / token.draftProb)
        const isAccepted = Math.random() < acceptProb || token.targetProb >= token.draftProb

        setTokens((prev) =>
          prev.map((t) =>
            t.id === token.id
              ? { ...t, status: 'verifying' }
              : t
          )
        )
        await new Promise((resolve) => setTimeout(resolve, 500))

        if (isAccepted) {
          setTokens((prev) =>
            prev.map((t) =>
              t.id === token.id
                ? { ...t, status: 'accepted' }
                : t
            )
          )
          accepted++
          setAcceptedCount(accepted)
          setCurrentPhase('accept')
        } else {
          setTokens((prev) =>
            prev.map((t) =>
              t.id === token.id
                ? { ...t, status: 'rejected' }
                : t
            )
          )
          setCurrentPhase('reject')
          break // 拒绝后停止
        }

        await new Promise((resolve) => setTimeout(resolve, 600))
      }

      setIsPlaying(false)
    }

    runDemo()
  }, [isPlaying, K, sampleTokens])

  const phaseInfo = {
    draft: {
      title: '阶段 1：Draft Model 推测生成',
      color: 'blue',
      description: `小模型快速生成 ${K} 个候选 tokens`,
    },
    verify: {
      title: '阶段 2：Target Model 批量验证',
      color: 'purple',
      description: '大模型一次性验证所有候选',
    },
    accept: {
      title: '阶段 3：接受 Token',
      color: 'green',
      description: 'target_prob ≥ draft_prob → 接受',
    },
    reject: {
      title: '阶段 3：拒绝 Token',
      color: 'red',
      description: 'target_prob < draft_prob → 拒绝并重新采样',
    },
  }

  const currentInfo = phaseInfo[currentPhase]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        Speculative Decoding 工作流程
      </h3>

      {/* 控制面板 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-2">
            推测长度 K：{K}
          </label>
          <input
            type="range"
            min="3"
            max="7"
            value={K}
            onChange={(e) => setK(parseInt(e.target.value))}
            disabled={isPlaying}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setIsPlaying(true)}
            disabled={isPlaying}
            className="flex-1 px-4 py-2 bg-green-500 text-white rounded-lg font-semibold hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ▶️ 开始演示
          </button>
          <button
            onClick={reset}
            className="flex-1 px-4 py-2 bg-gray-500 text-white rounded-lg font-semibold hover:bg-gray-600"
          >
            🔄 重置
          </button>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-sm text-green-700 dark:text-green-400">已接受 Tokens</p>
          <p className="text-2xl font-bold text-green-900 dark:text-green-200">
            {acceptedCount} / {K}
          </p>
        </div>
      </div>

      {/* 当前阶段信息 */}
      <div className={`mb-6 p-4 rounded-lg bg-${currentInfo.color}-50 dark:bg-${currentInfo.color}-900/20 border-2 border-${currentInfo.color}-300 dark:border-${currentInfo.color}-700`}>
        <h4 className={`text-lg font-bold text-${currentInfo.color}-900 dark:text-${currentInfo.color}-300 mb-2`}>
          {currentInfo.title}
        </h4>
        <p className={`text-sm text-${currentInfo.color}-700 dark:text-${currentInfo.color}-400`}>
          {currentInfo.description}
        </p>
      </div>

      {/* Token 可视化 */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-xl">
        <h4 className="text-lg font-bold text-gray-100 mb-4">
          生成的 Tokens
        </h4>

        <div className="space-y-3">
          <AnimatePresence>
            {tokens.map((token) => {
              const statusColor = {
                pending: 'gray',
                verifying: 'yellow',
                accepted: 'green',
                rejected: 'red',
              }[token.status]

              return (
                <motion.div
                  key={token.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className={`p-4 rounded-lg border-2 border-${statusColor}-300 dark:border-${statusColor}-600 bg-${statusColor}-50 dark:bg-${statusColor}-900/20`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="text-lg font-bold text-gray-100">
                        &quot;{token.text}&quot;
                      </span>
                      <span className={`text-xs px-2 py-1 rounded-full bg-${statusColor}-500 text-white font-bold`}>
                        {token.status === 'pending' && '等待验证'}
                        {token.status === 'verifying' && '验证中...'}
                        {token.status === 'accepted' && '✅ 接受'}
                        {token.status === 'rejected' && '❌ 拒绝'}
                      </span>
                    </div>

                    {(token.status === 'verifying' || token.status === 'accepted' || token.status === 'rejected') && (
                      <div className="flex gap-4 text-sm">
                        <div className="text-center">
                          <p className="text-xs text-gray-300">Draft Prob</p>
                          <p className="font-bold text-blue-700 dark:text-blue-300">
                            {token.draftProb.toFixed(2)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-300">Target Prob</p>
                          <p className="font-bold text-purple-700 dark:text-purple-300">
                            {token.targetProb.toFixed(2)}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-300">接受概率</p>
                          <p className={`font-bold ${
                            Math.min(1, token.targetProb / token.draftProb) >= 0.8
                              ? 'text-green-700 dark:text-green-300'
                              : 'text-red-700 dark:text-red-300'
                          }`}>
                            {Math.min(1, token.targetProb / token.draftProb).toFixed(2)}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {(token.status === 'accepted' || token.status === 'rejected') && (
                    <div className="mt-2 text-xs text-gray-300">
                      判定：min(1, {token.targetProb.toFixed(2)} / {token.draftProb.toFixed(2)}) = {Math.min(1, token.targetProb / token.draftProb).toFixed(2)}
                      {token.status === 'rejected' && ' → 拒绝后重新采样'}
                    </div>
                  )}
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      </div>

      {/* 架构对比 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
            Draft Model（小模型）
          </h5>
          <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
            <li>• 参数量：160M - 1B</li>
            <li>• 速度：10-20x 快于大模型</li>
            <li>• 作用：快速生成候选 tokens</li>
            <li>• 示例：LLaMA-160M</li>
          </ul>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <h5 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
            Target Model（大模型）
          </h5>
          <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
            <li>• 参数量：7B - 70B</li>
            <li>• 速度：标准速度</li>
            <li>• 作用：批量验证候选 tokens</li>
            <li>• 示例：LLaMA-7B / LLaMA-70B</li>
          </ul>
        </div>
      </div>

      {/* 性能提升 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-sm text-green-700 dark:text-green-400 mb-1">理论加速比</p>
          <p className="text-2xl font-bold text-green-900 dark:text-green-200">
            1 + α × K
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
            α=0.6, K=5 → 4.0x
          </p>
        </div>

        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-sm text-blue-700 dark:text-blue-400 mb-1">实际加速比</p>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-200">
            2.0 - 2.5x
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
            考虑小模型开销
          </p>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-sm text-purple-700 dark:text-purple-400 mb-1">额外成本</p>
          <p className="text-2xl font-bold text-purple-900 dark:text-purple-200">
            零
          </p>
          <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
            无需重新训练
          </p>
        </div>
      </div>

      {/* 公式说明 */}
      <div className="mt-6 p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">接受条件（Acceptance Criterion）：</p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{`对于每个候选 token t_i：
  1. 计算接受概率：p_accept = min(1, P_target(t_i) / P_draft(t_i))
  2. 采样：u ~ Uniform(0, 1)
  3. 如果 u < p_accept：接受 t_i
  4. 否则：拒绝 t_i，并从调整后的分布重新采样

保证：输出分布与标准自回归完全一致（数学证明见论文）`}
        </pre>
      </div>
    </div>
  )
}

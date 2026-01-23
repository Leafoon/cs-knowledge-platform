'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Zap, CheckCircle, XCircle } from 'lucide-react'

export default function SpeculativeDecodingFlow() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const maxSteps = 4
  const k = 3 // 推测token数

  useEffect(() => {
    if (isPlaying && step < maxSteps) {
      const timer = setTimeout(() => setStep(step + 1), 2000)
      return () => clearTimeout(timer)
    } else if (step >= maxSteps) {
      setIsPlaying(false)
    }
  }, [isPlaying, step])

  const reset = () => {
    setStep(0)
    setIsPlaying(false)
  }

  // 模拟draft model生成的tokens
  const draftTokens = ["is", "a", "field"]
  // 模拟验证结果（2个接受，1个拒绝）
  const accepted = [true, true, false]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Zap className="w-8 h-8 text-green-600" />
          <h3 className="text-2xl font-bold text-slate-800">Speculative Decoding 工作流程</h3>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={reset}
            className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 当前提示词 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <div className="text-sm text-slate-600 mb-1">当前输入</div>
        <div className="font-mono text-lg font-bold text-slate-800">"AI"</div>
        <div className="text-xs text-slate-500 mt-1">目标: 生成后续文本</div>
      </div>

      {/* 流程可视化 */}
      <div className="space-y-4 mb-6">
        {/* 步骤0: 初始状态 */}
        <AnimatePresence>
          {step >= 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white p-5 rounded-lg shadow-lg border-2 border-blue-200"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-blue-500 rounded-full text-white flex items-center justify-center font-bold">
                  0
                </div>
                <h4 className="font-bold text-blue-800">初始状态</h4>
              </div>
              <div className="ml-11 text-sm text-slate-700">
                输入序列: <span className="font-mono bg-blue-50 px-2 py-1 rounded">["AI"]</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 步骤1: Draft Model生成 */}
        <AnimatePresence>
          {step >= 1 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white p-5 rounded-lg shadow-lg border-2 border-green-200"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-green-500 rounded-full text-white flex items-center justify-center font-bold">
                  1
                </div>
                <h4 className="font-bold text-green-800">Draft Model 快速生成 {k} 个候选 token</h4>
                <div className="ml-auto text-xs bg-green-100 px-2 py-1 rounded text-green-700 font-bold">
                  快速（小模型）
                </div>
              </div>
              <div className="ml-11">
                <div className="text-sm text-slate-600 mb-2">使用小模型（如 LLaMA-7B）推测后续tokens:</div>
                <div className="flex gap-2">
                  {draftTokens.map((token, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.2 * idx }}
                      className="px-3 py-2 bg-green-100 border border-green-300 rounded font-mono"
                    >
                      "{token}"
                    </motion.div>
                  ))}
                </div>
                <div className="mt-2 text-xs text-slate-500">
                  候选序列: ["AI", "is", "a", "field"]
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 步骤2: Target Model并行验证 */}
        <AnimatePresence>
          {step >= 2 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white p-5 rounded-lg shadow-lg border-2 border-purple-200"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-purple-500 rounded-full text-white flex items-center justify-center font-bold">
                  2
                </div>
                <h4 className="font-bold text-purple-800">Target Model 并行验证所有候选</h4>
                <div className="ml-auto text-xs bg-purple-100 px-2 py-1 rounded text-purple-700 font-bold">
                  一次前向传播
                </div>
              </div>
              <div className="ml-11">
                <div className="text-sm text-slate-600 mb-2">使用大模型（如 LLaMA-70B）一次性验证:</div>
                <div className="bg-purple-50 p-3 rounded border border-purple-200">
                  <div className="font-mono text-sm">
                    logits = target_model(["AI", "is", "a", "field"])
                  </div>
                  <div className="text-xs text-purple-700 mt-1">
                    返回每个位置的概率分布，用于验证 draft tokens
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 步骤3: 逐个验证 */}
        <AnimatePresence>
          {step >= 3 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white p-5 rounded-lg shadow-lg border-2 border-orange-200"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-orange-500 rounded-full text-white flex items-center justify-center font-bold">
                  3
                </div>
                <h4 className="font-bold text-orange-800">逐个验证并采样</h4>
              </div>
              <div className="ml-11 space-y-2">
                {draftTokens.map((token, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 * idx }}
                    className={`p-3 rounded border-2 ${
                      accepted[idx]
                        ? 'bg-green-50 border-green-300'
                        : 'bg-red-50 border-red-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="font-mono font-bold">"{token}"</div>
                        {accepted[idx] ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600" />
                        )}
                      </div>
                      <div className={`text-sm font-bold ${
                        accepted[idx] ? 'text-green-700' : 'text-red-700'
                      }`}>
                        {accepted[idx] ? '✓ 接受' : '✗ 拒绝，重新采样'}
                      </div>
                    </div>
                    {!accepted[idx] && (
                      <div className="mt-2 text-xs text-slate-600 bg-white p-2 rounded">
                        从 target model 的概率分布重新采样 → &quot;branch&quot;
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 步骤4: 最终结果 */}
        <AnimatePresence>
          {step >= 4 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white p-5 rounded-lg shadow-lg border-2 border-blue-500"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-blue-600 rounded-full text-white flex items-center justify-center font-bold">
                  4
                </div>
                <h4 className="font-bold text-blue-800">生成结果</h4>
              </div>
              <div className="ml-11">
                <div className="text-sm text-slate-600 mb-2">接受的tokens:</div>
                <div className="flex gap-2 items-center mb-3">
                  <div className="font-mono bg-blue-50 px-3 py-2 rounded border border-blue-300">
                    &quot;AI&quot;
                  </div>
                  {accepted.map((isAccepted, idx) => (
                    <React.Fragment key={idx}>
                      <div className="text-slate-400">→</div>
                      <div className={`font-mono px-3 py-2 rounded border ${
                        isAccepted
                          ? 'bg-green-50 border-green-300'
                          : 'bg-orange-50 border-orange-300'
                      }`}>
                        {isAccepted ? `"${draftTokens[idx]}"` : '"branch"'}
                      </div>
                      {!isAccepted && <div className="text-xs text-red-600">(重采样)</div>}
                    </React.Fragment>
                  ))}
                </div>
                <div className="p-3 bg-blue-50 rounded border border-blue-200">
                  <div className="text-sm font-bold text-blue-800 mb-1">
                    本轮生成了 3 个 token（原本需要3次前向传播）
                  </div>
                  <div className="text-xs text-slate-600">
                    实际只用了 2 次前向传播（draft + target）
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 性能分析 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">性能分析</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-green-50 rounded border border-green-200">
            <div className="text-sm text-slate-600 mb-1">推测token数 (k)</div>
            <div className="text-3xl font-bold text-green-600">{k}</div>
          </div>
          <div className="p-4 bg-blue-50 rounded border border-blue-200">
            <div className="text-sm text-slate-600 mb-1">接受率 (α)</div>
            <div className="text-3xl font-bold text-blue-600">
              {((accepted.filter(a => a).length / accepted.length) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="p-4 bg-purple-50 rounded border border-purple-200">
            <div className="text-sm text-slate-600 mb-1">理论加速比</div>
            <div className="text-3xl font-bold text-purple-600">
              {((1 + k * (accepted.filter(a => a).length / accepted.length)) / (1 + k)).toFixed(2)}x
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-slate-50 rounded">
          <div className="text-sm font-bold text-slate-800 mb-2">加速比公式</div>
          <div className="font-mono text-sm text-slate-700">
            Speedup = (1 + k × α) / (1 + k)
          </div>
          <div className="text-xs text-slate-500 mt-1">
            k = 推测token数, α = 接受率（draft model与target model的一致性）
          </div>
        </div>

        <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded text-sm">
          <strong>关键因素</strong>: Draft model 应与 target model 架构相似，且在相似数据上训练，以提高接受率
        </div>
      </div>
    </div>
  )
}

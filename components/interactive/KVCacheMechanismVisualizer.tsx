'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Database } from 'lucide-react'

export default function KVCacheMechanismVisualizer() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [useCache, setUseCache] = useState(true)

  const prompt = "Hello world"
  const generated = ["!", " How", " are", " you", "?"]
  const maxSteps = generated.length

  useEffect(() => {
    if (isPlaying && step < maxSteps) {
      const timer = setTimeout(() => setStep(step + 1), 1200)
      return () => clearTimeout(timer)
    } else if (step >= maxSteps) {
      setIsPlaying(false)
    }
  }, [isPlaying, step, maxSteps])

  const reset = () => {
    setStep(0)
    setIsPlaying(false)
  }

  const currentSequence = prompt + generated.slice(0, step).join('')
  const computations = useCache ? step + 1 : (step + 1) * (step + 2) / 2

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Database className="w-8 h-8 text-purple-600" />
          <h3 className="text-2xl font-bold text-slate-800">KV Cache 工作机制</h3>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useCache}
              onChange={() => setUseCache(!useCache)}
              className="w-4 h-4"
            />
            <span className="text-sm font-medium text-slate-700">启用 KV Cache</span>
          </label>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
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

      {/* 当前状态 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-slate-600 mb-1">当前生成步骤</div>
            <div className="text-2xl font-bold text-purple-600">{step} / {maxSteps}</div>
          </div>
          <div>
            <div className="text-sm text-slate-600 mb-1">当前序列</div>
            <div className="font-mono text-lg font-bold text-slate-800">"{currentSequence}"</div>
          </div>
          <div>
            <div className="text-sm text-slate-600 mb-1">计算次数</div>
            <div className={`text-2xl font-bold ${useCache ? 'text-green-600' : 'text-red-600'}`}>
              {computations}
            </div>
          </div>
        </div>
      </div>

      {/* 可视化 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 不使用Cache */}
        <div className={`p-5 rounded-lg border-2 ${!useCache ? 'border-red-500 bg-red-50' : 'border-slate-200 bg-white'}`}>
          <h4 className="font-bold text-slate-800 mb-3">❌ 不使用 KV Cache</h4>
          <div className="space-y-2">
            {Array.from({ length: step + 1 }).map((_, idx) => (
              <motion.div
                key={`no-cache-${idx}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="p-2 bg-red-100 rounded border border-red-300"
              >
                <div className="text-xs text-slate-600 mb-1">步骤 {idx + 1}</div>
                <div className="flex gap-1">
                  {currentSequence.split('').slice(0, prompt.length + idx).map((char, charIdx) => (
                    <div
                      key={charIdx}
                      className="w-6 h-6 bg-red-400 rounded flex items-center justify-center text-white text-xs"
                    >
                      {char === ' ' ? '·' : char}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-red-700 mt-1">
                  重新计算所有 {prompt.length + idx} 个token的K、V
                </div>
              </motion.div>
            ))}
          </div>
          <div className="mt-3 p-3 bg-red-200 rounded">
            <div className="text-sm font-bold text-red-800">
              总计算量: O(n²) = {computations} 次前向传播
            </div>
          </div>
        </div>

        {/* 使用Cache */}
        <div className={`p-5 rounded-lg border-2 ${useCache ? 'border-green-500 bg-green-50' : 'border-slate-200 bg-white'}`}>
          <h4 className="font-bold text-slate-800 mb-3">✅ 使用 KV Cache</h4>
          
          {/* 初始prompt */}
          <div className="mb-3 p-2 bg-blue-100 rounded border border-blue-300">
            <div className="text-xs text-slate-600 mb-1">初始 Prompt</div>
            <div className="flex gap-1 mb-1">
              {prompt.split('').map((char, idx) => (
                <div
                  key={idx}
                  className="w-6 h-6 bg-blue-400 rounded flex items-center justify-center text-white text-xs"
                >
                  {char === ' ' ? '·' : char}
                </div>
              ))}
            </div>
            <div className="text-xs text-blue-700">计算并缓存 K、V</div>
          </div>

          {/* 缓存的KV */}
          {step > 0 && (
            <div className="mb-3 p-2 bg-purple-100 rounded border border-purple-300">
              <div className="text-xs text-purple-700 font-bold mb-1">📦 已缓存的 KV</div>
              <div className="flex gap-1">
                {currentSequence.split('').slice(0, -1).map((char, idx) => (
                  <div
                    key={idx}
                    className="w-6 h-6 bg-purple-400 rounded flex items-center justify-center text-white text-xs opacity-70"
                  >
                    {char === ' ' ? '·' : char}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 新生成的token */}
          {generated.slice(0, step).map((token, idx) => (
            <motion.div
              key={`cache-${idx}`}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-2 p-2 bg-green-100 rounded border border-green-300"
            >
              <div className="text-xs text-slate-600 mb-1">步骤 {idx + 1}</div>
              <div className="flex gap-1">
                <div className="w-6 h-6 bg-green-500 rounded flex items-center justify-center text-white text-xs font-bold">
                  {token === ' ' ? '·' : token}
                </div>
              </div>
              <div className="text-xs text-green-700 mt-1">
                只计算新token，复用缓存
              </div>
            </motion.div>
          ))}

          <div className="mt-3 p-3 bg-green-200 rounded">
            <div className="text-sm font-bold text-green-800">
              总计算量: O(n) = {step + 1} 次前向传播
            </div>
          </div>
        </div>
      </div>

      {/* 性能对比 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">性能对比</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-slate-50 rounded">
            <div className="text-sm text-slate-600 mb-1">不使用Cache</div>
            <div className="text-2xl font-bold text-red-600">O(n²)</div>
            <div className="text-xs text-slate-500 mt-1">每步重算所有token</div>
          </div>
          <div className="p-4 bg-green-50 rounded">
            <div className="text-sm text-slate-600 mb-1">使用Cache</div>
            <div className="text-2xl font-bold text-green-600">O(n)</div>
            <div className="text-xs text-slate-500 mt-1">每步只算新token</div>
          </div>
          <div className="p-4 bg-blue-50 rounded">
            <div className="text-sm text-slate-600 mb-1">加速比</div>
            <div className="text-2xl font-bold text-blue-600">
              {step > 0 ? ((step + 1) * (step + 2) / 2 / (step + 1)).toFixed(1) : 1}x
            </div>
            <div className="text-xs text-slate-500 mt-1">随序列长度增加</div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded text-sm text-slate-700">
          <strong>KV Cache 显存占用</strong>: 2 × batch × layers × heads × seq_len × head_dim × 2 bytes (FP16)
          <br />
          <span className="text-purple-700">示例: LLaMA-7B, seq_len=2048 → ~1 GB</span>
        </div>
      </div>
    </div>
  )
}

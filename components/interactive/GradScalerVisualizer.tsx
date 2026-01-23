'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, AlertCircle, CheckCircle } from 'lucide-react'

export default function GradScalerVisualizer() {
  const [scale, setScale] = useState(65536)
  const [step, setStep] = useState(0)
  const [history, setHistory] = useState<{step: number, scale: number, status: string}[]>([])

  const gradientScenarios = [
    { name: '极小梯度', original: 1.2e-7, description: 'FP16 会下溢为 0' },
    { name: '正常梯度', original: 0.003, description: '标准训练范围' },
    { name: '大梯度', original: 5.2, description: '可能需要裁剪' },
  ]

  const simulateStep = (hasNaN: boolean) => {
    const newStep = step + 1
    let newScale = scale
    let status = ''

    if (hasNaN) {
      // 出现 NaN，减半
      newScale = Math.max(scale / 2, 1)
      status = '检测到 NaN，scale ÷ 2'
    } else {
      // 正常步骤
      if (step > 0 && step % 2000 === 0) {
        // 每 2000 步翻倍
        newScale = Math.min(scale * 2, 65536 * 4)
        status = '连续 2000 步成功，scale × 2'
      } else {
        status = '正常训练'
      }
    }

    setScale(newScale)
    setStep(newStep)
    setHistory([...history, { step: newStep, scale: newScale, status }].slice(-5))
  }

  const reset = () => {
    setScale(65536)
    setStep(0)
    setHistory([])
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <TrendingUp className="w-8 h-8 text-indigo-600" />
        <h3 className="text-2xl font-bold text-slate-800">GradScaler 动态缩放机制</h3>
      </div>

      {/* 当前状态 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-5 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">当前 Scale Factor</div>
          <div className="text-3xl font-bold text-indigo-600">{scale.toLocaleString()}</div>
          <div className="text-xs text-slate-500 mt-1">
            = 2<sup>{Math.log2(scale)}</sup>
          </div>
        </div>

        <div className="bg-white p-5 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">训练步数</div>
          <div className="text-3xl font-bold text-blue-600">{step}</div>
          <div className="text-xs text-slate-500 mt-1">
            {step < 2000 ? `距离下次增大还有 ${2000 - (step % 2000)} 步` : '周期完成'}
          </div>
        </div>

        <div className="bg-white p-5 rounded-lg shadow">
          <div className="text-sm text-slate-600 mb-1">状态</div>
          <div className="text-lg font-bold text-green-600 flex items-center gap-2">
            <CheckCircle className="w-5 h-5" />
            正常
          </div>
        </div>
      </div>

      {/* 梯度缩放演示 */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <h4 className="font-bold text-slate-800 mb-4">梯度缩放效果（Scale = {scale}）</h4>
        <div className="space-y-3">
          {gradientScenarios.map((scenario, idx) => {
            const scaled = scenario.original * scale
            const isUnderflow = scenario.original < 6.1e-5
            const isSafe = scaled >= 6.1e-5

            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-4 border-2 border-slate-200 rounded-lg"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-slate-700">{scenario.name}</div>
                  <div className="text-xs text-slate-500">{scenario.description}</div>
                </div>

                <div className="flex items-center gap-4">
                  {/* 原始梯度 */}
                  <div className="flex-1">
                    <div className="text-xs text-slate-600 mb-1">原始梯度</div>
                    <div className={`p-2 rounded font-mono text-sm ${
                      isUnderflow ? 'bg-red-50 text-red-700' : 'bg-slate-50 text-slate-700'
                    }`}>
                      {scenario.original.toExponential(2)}
                    </div>
                    {isUnderflow && (
                      <div className="text-xs text-red-600 mt-1">⚠️ FP16 下溢为 0</div>
                    )}
                  </div>

                  <div className="text-2xl text-slate-400">×</div>

                  {/* Scale */}
                  <div className="flex-1">
                    <div className="text-xs text-slate-600 mb-1">Scale</div>
                    <div className="p-2 bg-indigo-50 rounded font-mono text-sm text-indigo-700">
                      {scale.toLocaleString()}
                    </div>
                  </div>

                  <div className="text-2xl text-slate-400">=</div>

                  {/* 缩放后梯度 */}
                  <div className="flex-1">
                    <div className="text-xs text-slate-600 mb-1">缩放后梯度</div>
                    <div className={`p-2 rounded font-mono text-sm ${
                      isSafe ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'
                    }`}>
                      {scaled.toFixed(4)}
                    </div>
                    {isSafe && isUnderflow && (
                      <div className="text-xs text-green-600 mt-1">✓ 安全范围</div>
                    )}
                  </div>
                </div>

                {/* 进度条 */}
                <div className="mt-3 h-2 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min((scaled / 100) * 100, 100)}%` }}
                    className={`h-full ${
                      isSafe ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* 模拟按钮 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <button
          onClick={() => simulateStep(false)}
          className="px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold"
        >
          ✓ 正常训练步
        </button>
        <button
          onClick={() => simulateStep(true)}
          className="px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold"
        >
          ✗ 出现 NaN
        </button>
        <button
          onClick={reset}
          className="px-4 py-3 bg-slate-600 text-white rounded-lg hover:bg-slate-700"
        >
          重置
        </button>
      </div>

      {/* 历史记录 */}
      {history.length > 0 && (
        <div className="bg-white p-5 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-3">训练历史（最近 5 步）</h4>
          <div className="space-y-2">
            {history.map((entry, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 bg-slate-50 rounded text-sm"
              >
                <span className="font-mono text-slate-600">Step {entry.step}</span>
                <span className="font-mono text-indigo-600">Scale: {entry.scale}</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  entry.status.includes('NaN')
                    ? 'bg-red-100 text-red-700'
                    : entry.status.includes('×')
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-green-100 text-green-700'
                }`}>
                  {entry.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 说明 */}
      <div className="mt-6 p-5 bg-yellow-50 border border-yellow-300 rounded-lg">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
          <div className="text-sm text-slate-700">
            <strong>动态调整策略</strong>：
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>初始 scale = 65536 (2<sup>16</sup>)</li>
              <li>检测到 <code className="bg-white px-1 rounded">inf/NaN</code> → scale ÷ 2（避免梯度爆炸）</li>
              <li>连续 2000 步无异常 → scale × 2（提高精度）</li>
              <li>scale 范围：[1, 262144]（自动限制）</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

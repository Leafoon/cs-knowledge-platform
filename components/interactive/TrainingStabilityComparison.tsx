'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, AlertTriangle, CheckCircle } from 'lucide-react'

export default function TrainingStabilityComparison() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  // 模拟训练 loss 数据
  const generateLossData = (precision: 'FP32' | 'BF16' | 'FP16', steps: number) => {
    const data: { step: number; loss: number; hasNaN: boolean }[] = []
    let baseLoss = 3.5

    for (let i = 0; i <= steps; i++) {
      // 基础下降
      baseLoss = baseLoss * 0.9985

      let loss = baseLoss
      let hasNaN = false

      if (precision === 'FP32') {
        // FP32 最稳定
        loss += (Math.random() - 0.5) * 0.05
      } else if (precision === 'BF16') {
        // BF16 轻微波动
        loss += (Math.random() - 0.5) * 0.06
      } else {
        // FP16 波动大，偶现 NaN
        loss += (Math.random() - 0.5) * 0.12
        if (i > 50 && Math.random() < 0.05) {  // 5% 概率 NaN
          hasNaN = true
          loss = NaN
        }
      }

      data.push({ step: i, loss, hasNaN })
    }

    return data
  }

  const fp32Data = generateLossData('FP32', 100)
  const bf16Data = generateLossData('BF16', 100)
  const fp16Data = generateLossData('FP16', 100)

  const getCurrentLoss = (data: typeof fp32Data) => {
    return data[Math.min(step, data.length - 1)]
  }

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setStep((prev) => {
          if (prev >= 100) {
            setIsPlaying(false)
            return 100
          }
          return prev + 1
        })
      }, 50)
      return () => clearInterval(interval)
    }
  }, [isPlaying])

  const fp32Current = getCurrentLoss(fp32Data)
  const bf16Current = getCurrentLoss(bf16Data)
  const fp16Current = getCurrentLoss(fp16Data)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-rose-600" />
          <h3 className="text-2xl font-bold text-slate-800">训练稳定性对比</h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => {
              setStep(0)
              setIsPlaying(true)
            }}
            disabled={isPlaying}
            className="px-4 py-2 bg-rose-600 text-white rounded-lg hover:bg-rose-700 disabled:bg-slate-400"
          >
            {isPlaying ? '播放中...' : '开始训练'}
          </button>
          <button
            onClick={() => {
              setStep(0)
              setIsPlaying(false)
            }}
            className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700"
          >
            重置
          </button>
        </div>
      </div>

      {/* 步数指示 */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-slate-700">训练步数</span>
          <span className="font-mono text-lg font-bold text-rose-600">{step} / 100</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            animate={{ width: `${step}%` }}
            className="h-full bg-rose-500"
          />
        </div>
      </div>

      {/* Loss 曲线图 */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <h4 className="font-bold text-slate-800 mb-4">Loss 曲线对比</h4>
        
        <div className="relative h-64 border-2 border-slate-200 rounded-lg p-4 bg-slate-50">
          {/* Y 轴标签 */}
          <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-slate-600 pr-2">
            <span>3.5</span>
            <span>3.2</span>
            <span>2.9</span>
            <span>2.6</span>
            <span>2.3</span>
          </div>

          {/* 绘图区域 */}
          <svg className="w-full h-full ml-8" viewBox="0 0 400 200">
            {/* FP32 曲线（蓝色） */}
            <path
              d={fp32Data
                .slice(0, step + 1)
                .map((d, i) => {
                  const x = (i / 100) * 400
                  const y = 200 - ((d.loss - 2.3) / 1.2) * 200
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                })
                .join(' ')}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
            />

            {/* BF16 曲线（绿色） */}
            <path
              d={bf16Data
                .slice(0, step + 1)
                .map((d, i) => {
                  const x = (i / 100) * 400
                  const y = 200 - ((d.loss - 2.3) / 1.2) * 200
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                })
                .join(' ')}
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
            />

            {/* FP16 曲线（橙色，可能中断） */}
            {fp16Data.slice(0, step + 1).map((d, i) => {
              if (d.hasNaN) return null
              const x = (i / 100) * 400
              const y = 200 - ((d.loss - 2.3) / 1.2) * 200
              const nextPoint = fp16Data[i + 1]
              if (!nextPoint || nextPoint.hasNaN) {
                return (
                  <circle key={i} cx={x} cy={y} r="3" fill="#f59e0b" />
                )
              }
              const nextX = ((i + 1) / 100) * 400
              const nextY = 200 - ((nextPoint.loss - 2.3) / 1.2) * 200
              return (
                <line
                  key={i}
                  x1={x}
                  y1={y}
                  x2={nextX}
                  y2={nextY}
                  stroke="#f59e0b"
                  strokeWidth="2"
                />
              )
            })}

            {/* NaN 标记 */}
            {fp16Data.slice(0, step + 1).map((d, i) => {
              if (!d.hasNaN) return null
              const x = (i / 100) * 400
              return (
                <g key={`nan-${i}`}>
                  <circle cx={x} cy="100" r="5" fill="#ef4444" />
                  <text x={x} y="90" fontSize="10" fill="#ef4444" textAnchor="middle">
                    NaN
                  </text>
                </g>
              )
            })}
          </svg>

          {/* X 轴标签 */}
          <div className="absolute bottom-0 left-8 right-0 flex justify-between text-xs text-slate-600 mt-2">
            <span>0</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
          </div>
        </div>

        {/* 图例 */}
        <div className="flex justify-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-1 bg-blue-500"></div>
            <span className="text-sm text-slate-700">FP32 (平滑)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-1 bg-green-500"></div>
            <span className="text-sm text-slate-700">BF16 (轻微波动)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-1 bg-orange-500"></div>
            <span className="text-sm text-slate-700">FP16 (波动大)</span>
          </div>
        </div>
      </div>

      {/* 当前状态 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* FP32 */}
        <div className="bg-white p-5 rounded-lg shadow border-t-4 border-blue-500">
          <div className="flex items-center justify-between mb-3">
            <h5 className="font-bold text-blue-700">FP32</h5>
            <CheckCircle className="w-5 h-5 text-green-600" />
          </div>
          <div className="text-3xl font-bold text-blue-600 mb-2">
            {fp32Current.loss.toFixed(3)}
          </div>
          <div className="text-xs text-slate-600">
            方差: ±0.05 (最稳定)
          </div>
        </div>

        {/* BF16 */}
        <div className="bg-white p-5 rounded-lg shadow border-t-4 border-green-500">
          <div className="flex items-center justify-between mb-3">
            <h5 className="font-bold text-green-700">BF16</h5>
            <CheckCircle className="w-5 h-5 text-green-600" />
          </div>
          <div className="text-3xl font-bold text-green-600 mb-2">
            {bf16Current.loss.toFixed(3)}
          </div>
          <div className="text-xs text-slate-600">
            方差: ±0.06 (稳定)
          </div>
        </div>

        {/* FP16 */}
        <div className={`bg-white p-5 rounded-lg shadow border-t-4 ${
          fp16Current.hasNaN ? 'border-red-500' : 'border-orange-500'
        }`}>
          <div className="flex items-center justify-between mb-3">
            <h5 className="font-bold text-orange-700">FP16</h5>
            {fp16Current.hasNaN ? (
              <AlertTriangle className="w-5 h-5 text-red-600" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-600" />
            )}
          </div>
          <div className={`text-3xl font-bold mb-2 ${
            fp16Current.hasNaN ? 'text-red-600' : 'text-orange-600'
          }`}>
            {fp16Current.hasNaN ? 'NaN' : fp16Current.loss.toFixed(3)}
          </div>
          <div className="text-xs text-slate-600">
            方差: ±0.12 (波动大)
          </div>
        </div>
      </div>

      {/* 稳定性统计 */}
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h4 className="font-bold text-slate-800 mb-4">100 次独立训练成功率</h4>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-slate-700">FP32</span>
            <div className="flex items-center gap-2">
              <div className="w-64 h-8 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 flex items-center justify-center text-white text-sm font-bold" style={{ width: '100%' }}>
                  100%
                </div>
              </div>
              <span className="font-mono text-blue-600 font-bold">100/100</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-700">BF16</span>
            <div className="flex items-center gap-2">
              <div className="w-64 h-8 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 flex items-center justify-center text-white text-sm font-bold" style={{ width: '100%' }}>
                  100%
                </div>
              </div>
              <span className="font-mono text-green-600 font-bold">100/100</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-slate-700">FP16</span>
            <div className="flex items-center gap-2">
              <div className="w-64 h-8 bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-orange-500 flex items-center justify-center text-white text-sm font-bold" style={{ width: '95%' }}>
                  95%
                </div>
              </div>
              <span className="font-mono text-orange-600 font-bold">95/100</span>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-slate-700">
          <strong>FP16 失败原因</strong>: 5 次训练出现 NaN（梯度下溢或爆炸），需重启训练
        </div>
      </div>

      {/* 总结 */}
      <div className="mt-6 p-5 bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-300 rounded-lg">
        <h4 className="font-bold text-green-800 mb-2">稳定性排序</h4>
        <div className="text-slate-700 text-sm">
          <strong>FP32 ≈ BF16</strong> &gt;&gt; <strong>FP16</strong>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>BF16 稳定性接近 FP32（动态范围相同）</li>
            <li>FP16 有 ~5% 概率出现 NaN（需监控 + loss scaling）</li>
            <li>生产环境推荐：<strong>BF16</strong>（速度与稳定性最优）</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
